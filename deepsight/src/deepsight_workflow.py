import os
import re
import time
import subprocess
import requests
import duckdb
import numpy as np
import luigi
import polars as pl
import pdb

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import fs
from genative import APClient, wafer
from src.base import DuckdbBase, DataNotExists, CliBase

from memory_profiler import profile
import gc


def clean_column_name(column_name: str) -> str:
    column_name = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", column_name)
    column_name = column_name.lower()
    for c in [" ", "-", "@", ".", "(", ")", "[", "]"]:
        replace_with = "_set" if c == "@" else "_"
        column_name = column_name.replace(c, replace_with)
    column_name = re.sub(r"_{2,}", "_", column_name).rstrip("_")
    column_name = re.sub(r"_[a-z0-9_]+_$", "", column_name)
    return column_name


class KnimeLoader(CliBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = rf"{self.data_root}/raw_data/tick_data/year={self.year}\month={self.month}\day={self.day}"
        self.api_url = os.getenv("SMART_API", "http://rushi.asia.tel.com:9090/")
        self.knime_path = os.getenv("KNIME_PATH", "D:/KNIME/ENV/5.3.2_20250228/knime/")

    @CliBase.success_output
    def run(self):
        from_date=self._extraction_date.replace(minute=0, second=0, microsecond=0)
        to_date=self._extraction_date.replace(minute=59, second=59, microsecond=999999)
        command = [
            f"{self.knime_path}/knime.exe",
            "-consoleLog",
            "-nosplash",
            "-reset",
            "-nosave",
            "--launcher.suppressErrors",
            "-application org.knime.product.KNIME_BATCH_APPLICATION",
            # f"""-preferences={os.path.abspath("knime/DataExtractor.epf")}""",
            f"-workflowFile={os.path.join(self.project_root, "src/knime/DataExtractor_5.2.1.knwf")}",
            f"""-workflow.variable=out_folder,"{self.data_root.replace("\\","/")}raw_data/",String""",
            f"""-workflow.variable=from_time,"{from_date.strftime("%Y-%m-%d %H:%M:%S")}",String""", # YYYY-MM-DD HH
            f"""-workflow.variable=to_time,"{to_date.strftime("%Y-%m-%d %H:%M:%S")}",String""", # YYYY-MM-DD HH
            f"""-workflow.variable=api_url,"{self.api_url}",String"""
        ]
        self.logger.info(f"{self.custom_task_id} started")
        self.logger.info(f"{" ".join(command)}")
        
        process = subprocess.run(' '.join(command), capture_output=True, text=True)
        if process.returncode != 0:
            self.logger.error(f"{process.stderr}")
            raise RuntimeError(f"[DataExtractor] ERROR: \n{process.stderr}")
        else:
            self.logger.info(f"{self.custom_task_id} finished")
            return self.custom_task_id
        
        # process = subprocess.Popen(command, stdout=subprocess.PIPE)
        # process.wait()
        # return self.custom_task_id


class DatApClient(APClient):
    def tools_and_chambers(
        self,
        **kwargs,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        v: pl.LazyFrame = self.tool_chamber_tree_data(**kwargs)

        tool_list = v.filter(pl.col("type").eq("tool") | pl.col("type").eq("importedtool")).select(
            pl.col("text").alias("Tool"), pl.col("id")
        )

        chamber_list = (
            v.filter(pl.col("type").eq("chamber"))
            .join(
                v.filter(pl.col("type").eq("tool")|pl.col("type").eq("importedtool")).select(
                    pl.col("text").alias("mainframe"), "id"
                ),
                left_on="parent",
                right_on="id",
                how="left",
            )
            .with_columns(
                chamber=pl.format(
                    "{}/{}",
                    pl.col("mainframe"),
                    pl.col("text"),
                )
            )
            .select(
                "id",
                pl.col("chamber").alias("Tool/Chamber"),
            )
        )

        return tool_list, chamber_list


class TactrasLoader(DuckdbBase):
    tool_type = luigi.Parameter(default="tactras")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url = os.getenv("SMART_API", "http://127.0.0.1:9000")
        self.data_path = Path(self.data_root) / "raw_data" / "tick_data" / f"year={self.year}" / f"month={int(self.month):02d}" / f"day={int(self.day):02d}"
        self.tool_type = self.tool_type
        self.chunk_size = 1000
        self.apc_client = self.get_apc_client(self.tool_type, self.api_url)
        pl.Config.set_fmt_str_lengths(150)

    def get_apc_client(self, tool_type, host_url, as_pandas=False):
        return DatApClient(tool_type=tool_type, as_pandas=as_pandas, host_url=host_url)

    def wafer_list(self, chamber_list: List[str]) -> pl.DataFrame:
        self.logger.debug(f"Fetching wafer list for chambers: {chamber_list}")
        wafers = wafer.waferlist(
            self.apc_client,
            chambers=chamber_list,
            # from_date=self._extraction_date.replace(minute=0, second=0, microsecond=0),
            # to_date=self._extraction_date.replace(minute=59, second=59, microsecond=999999),
            from_date=datetime(int(self.year), int(self.month), int(self.day), int(self.hour), 0,0),
            to_date=datetime(int(self.year), int(self.month), int(self.day), int(self.hour), 59,59)
        ).collect()
        self.logger.debug(f"Wafer list fetched with {len(wafers)} records")
        return wafers

    def _check_jobs_finished(self, jobids: List[str], sleep_sec: int = 5, timeout_sec: int = 300) -> pl.DataFrame:
        self.logger.info(f"Checking job status for {len(jobids)} jobids with timeout={timeout_sec}s")
        start_time = time.time()
        while True:
            status = wafer.get_job_status(self.apc_client, np.array(jobids)).with_columns(
                pl.when(pl.col("jobStatus").is_in(["completed", "failed"]))
                .then(True)
                .otherwise(False)
                .alias("is_finished")
            )
            self.logger.debug(f"Job statuses:\n{status}")
            if status["is_finished"].all():
                self.logger.info("All jobs finished")
                return status
            if time.time() - start_time > timeout_sec:
                raise TimeoutError("Timeout while waiting for jobs to finish")
            self.logger.debug(f"Jobs not finished yet, sleeping {sleep_sec}s...")
            time.sleep(sleep_sec)

    def wait_export(self, jobids: List[str]) -> pl.DataFrame:
        self.logger.info("Waiting for export jobs to complete...")
        return self._check_jobs_finished(jobids, sleep_sec=5, timeout_sec=600)

    def wait_download(self, jobids: List[str]) -> pl.DataFrame:
        self.logger.info("Waiting for download jobs to complete...")
        return self._check_jobs_finished(jobids, sleep_sec=1, timeout_sec=300)
    
    def get_sensor_list(self, ids: List[str]) -> List[Dict[str, Any]]:
        sensor_info: List[Dict[str, Any]] = []
        def fetch_sensor_list(id_: str) -> List[Dict[str, Any]]:
            try:
                response = requests.post(
                    f"{self.api_url}/dialog/SensorSelecterDialog/getSensorList",
                    data={
                        "tool": [f"{id_}"],
                        "undisclosedParameters": "hidden", # None (other option)
                        "isSelectedWaferForPlasma": False,
                        "addGasLineNameForDisplay": False,
                        "step": ["-1"],
                        "addPlasmaParameter": False,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                records = response.json()
                return records
            except (requests.RequestException, ValueError) as e:
                self.logger.error(f"Failed to get sensor list for id {id_}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_id = {executor.submit(fetch_sensor_list, id_): id_ for id_ in ids}
            for future in as_completed(future_to_id):
                id_ = future_to_id[future]
                try:
                    records = future.result()
                    for record in records:
                        sensors = [f"{sensor['text']}/{sensor['isHSM']}" for sensor in record.get("member", [])]
                        sensor_info.append(
                            {"id": id_, "label": record.get("label", ""), "10ms": record.get("isHSM"), "sensors": sensors}
                        )
                except Exception:
                    self.logger.error(f"Exception occurred fetching sensor list for {id_}", exc_info=True)
                    continue

        self.logger.debug(f"Total sensor info entries fetched: {len(sensor_info)}")
        return sensor_info

    def split_filename(self, name: str, value: pl.LazyFrame) -> Dict[str, Any]:
        pattern = re.compile(r"^tickdata_(\d{14})_([A-Za-z0-9.\s]+?)(?:_(\d+))?_TEL ([^_]+)_([^_]+)_([^_]+)\.csv$")
        match = pattern.match(name)
        if not match:
            self.logger.warning(f"Can't parsing name : {name}")
            return {"start_time": None}

        return {
            "start_time": match.group(1),
            "pjid": match.group(2).replace("_", "/"),
            "wafer_id": match.group(4),
            "load_port": match.group(5),
            "slot": match.group(6),
            "tick_data": value,
        }

    def clean_null_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        n_rows = df.height
        all_null_cols = [col for col in df.columns if df.select(pl.col(col).null_count()).item() == n_rows]
        return df.drop(all_null_cols)

    def export_tick(self, jids: List[str], sensor_list: List[str], wafer_list: pl.DataFrame) -> Union[str, None]:
        # TODO : Remove test limits before production
        jids_limited = jids  # 전체 사용
        sensors_limited = sensor_list#[:800]
        # sensors_limited = ['LF2 Pulse Frequency']
        limit_jobids = wafer.export_tickdata(self.apc_client, jids=jids_limited[:2], sensor_names=sensors_limited)
        tick_data_df = self.download_and_parse_tickdata(limit_jobids)
        if tick_data_df.is_empty() or len(tick_data_df) == 0:
            self.logger.info("Tick data DataFrame is empty.")
            return None

        result = self.process_and_save_tickdata(wafer_list, tick_data_df)

        result = self.clean_null_cols(result)

        meta_cols = {
            "time", "step", "sequence", "tool/chamber", "start_time", "end_time", "system_recipe", "process_recipe",
            "contents", "lotid", "cjid", "pjid", "carrier_id", "wafer_id", "slot", "load_port", "wafer_type",
            "processing_order_in_lot", "processing_order_per_chamber", "recipe_type", "recipe_status", "idle_time",
            "jid", "has_dummy_wafer", "id", "file_name", "year", "month", "day", "hour"
        }

        sensor_cols = list(set(result.columns) - meta_cols)

        self.logger.info(f"Exporting tick data for {len(jids_limited)} jobids and {len(sensor_cols)} sensors")

        return wafer.export_tickdata(self.apc_client, jids=jids_limited, sensor_names=sensor_cols)

    def get_sensor_list_filter(self, chamber_sensors: pl.DataFrame, chamber_id: str) -> List[str]:
        filtered = (
            chamber_sensors.filter(
                (pl.col("Tool/Chamber") == chamber_id) & (pl.col("label") != "OES_n") & (pl.col("label") != "OES")
            )
            .explode("sensors")
            .with_columns(
                pl.col("sensors").str.split("/").list.get(0).alias("sensor"),
                pl.col("sensors").str.split("/").list.get(1).alias("isHSM"),
            )
            .filter(pl.col("isHSM") == "False")
            .filter(
                (~pl.col("sensor").str.contains("(10ms)")) & \
                    (~pl.col("sensor").str.contains("@")) & \
                        (~pl.col("sensor").str.contains("setting data")) & \
                            (~pl.col("sensor").str.contains("STEP")) & \
                                (~pl.col("sensor").str.contains("LF2 Pulse Frequency"))
            )
            .select("sensor")
        )
        sensors = filtered.unique(maintain_order=True).to_series().to_list()
        self.logger.info(f"Filtered sensors for chamber {chamber_id}: {len(sensors)} sensors")

        return sensors

    def unzip_trace(self, filepath: Union[str, Path]) -> Dict[str, pl.LazyFrame]:
        self.logger.debug(f"Unzipping and loading trace from {filepath}")
        try:
            return fs.ticktrace.load_every_trace_zip(str(filepath), contents_start_from=15)
        except Exception:
            self.logger.error(f"Failed to unzip trace file {filepath}", exc_info=True)
    
            return {}

    def load_chamber_and_sensors(self) -> (pl.DataFrame, pl.DataFrame):
        self.logger.info("Loading chamber data and sensor list")
        chamber_df = (
            self.apc_client.chamber_df.filter(pl.col("Tool/Chamber").is_not_null())
            .with_columns(pl.col("Tool/Chamber").str.split("/").list.get(0).alias("prefix"))
            .collect()
        )
        unique_tool_df = chamber_df.unique(subset=["prefix"])

        chamber_ids = unique_tool_df["id"].to_list()
        
        sensor_parquet_path = Path(self.data_root) / "sensor_list.parquet"
        if not sensor_parquet_path.exists():
            sensor_data = self.get_sensor_list(chamber_ids)
            sensor_df = pl.LazyFrame(sensor_data) if sensor_data else pl.LazyFrame([])
            # 조인 시 컬렉트(실행) 신중히 처리
            chamber_sensors = (
                unique_tool_df.select(["id", "prefix"])
                .join(sensor_df.collect(), on="id", how="left")
                .join(chamber_df, on="prefix", how="inner")
            )
            chamber_sensors.write_parquet(str(sensor_parquet_path))
        else:
            chamber_sensors = pl.read_parquet(str(sensor_parquet_path))
        self.logger.debug(f"Loaded chamber_sensors with {len(chamber_sensors)} rows")
        return chamber_df, chamber_sensors

    def export_tickdata_for_chambers(self, chamber_sensors: pl.DataFrame, wafer_list: pl.DataFrame) -> (List[str], List[Dict[str, Any]]):
        jobids = []
        chamber_sensors_dict = []

        unique_chambers = chamber_sensors["Tool/Chamber"].unique().to_list()
        self.logger.info(f"Exporting tick data for {len(unique_chambers)} chambers")

        for chamber_id in unique_chambers:
            jids = wafer_list.filter(pl.col("tool/chamber") == chamber_id).select("jid").to_series().to_list()
            self.logger.info(f"Chamber {chamber_id} has {len(jids)} jids")
            if not jids:
                continue

            sensors = self.get_sensor_list_filter(chamber_sensors, chamber_id)
            if not sensors:
                self.logger.warning(f"No sensors found for chamber {chamber_id}, skipping export")
                continue

            chamber_sensors_dict.append({"chamber_id": chamber_id, "sensor_list": sensors})

            # jobid = self.export_tick(jids, sensors, wafer_list)
            jobid = wafer.export_tickdata(self.apc_client, jids=jids, sensor_names=sensors)
            if jobid is None:
                self.logger.warning("Job ID is None - skipping this chamber")
                continue
            #TODO(Kyle) Option으로 개수 제한 걸도록 수정 필요
            jobids.append(jobid)

        self.logger.info(f"Total {len(jobids)} jobids exported")
        return jobids, chamber_sensors_dict
    
    @profile
    def download_and_parse_tickdata(self, jobids: List[str]) -> pl.DataFrame:
        self.logger.info("Waiting for export jobs to finish")
        try:
            self.wait_export(jobids)
        except TimeoutError as e:
            self.logger.error(f"Timeout occurred while waiting for export jobs: {e}")
            return pl.DataFrame()

        self.logger.info("Downloading tick data files")
        try:
            downloads_df = wafer.download(self.apc_client, np.array(jobids), output_path=Path("./tmp"))
        except Exception:
            self.logger.error("Failed to download tick data", exc_info=True)
            return pl.DataFrame()

        downloaded_files = downloads_df["download"].to_list()
        self.logger.info(f"Downloaded {len(downloaded_files)} tick data files")

        data_dict = {}
        for filepath in downloaded_files:
            extracted = self.unzip_trace(filepath)
            data_dict.update(extracted)
            del extracted

        splited_data = [{"file_name": filename, "tick_data": value} for filename, value in data_dict.items()]
        self.logger.debug(f"Parsed {len(splited_data)} tick data entries")
        del data_dict
        gc.collect()

        return pl.DataFrame(splited_data)
    
    @profile
    def process_and_save_tickdata(self, wafer_list: pl.DataFrame, tick_data_df: pl.DataFrame, save: bool = False) -> Union[pl.DataFrame, None]:
        final_df = wafer_list.join(tick_data_df, on=["file_name"])
        self.logger.info(f"Processing and saving tick data for {len(final_df)} records")
        all_lazyframes = []
        # self.logger.info(f"{final_df}")
        for row in final_df.iter_rows(named=True):
            tick_lazy: pl.LazyFrame = row["tick_data"]
            value_columns = set(tick_lazy.columns) - {"Time", "Step", "Sequence"}

            all_null_expr = pl.all_horizontal([pl.col(c).is_null() for c in value_columns])
            tick_lazy = tick_lazy.with_columns(all_null_expr.alias("all_null")).filter(~pl.col("all_null")).drop("all_null")

            extra_cols = {k: v for k, v in row.items() if k != "tick_data"}
            extra_exprs = [pl.lit(v).alias(k) for k, v in extra_cols.items()]
            partition_exprs = [
                pl.lit(self._extraction_date.strftime("%Y")).alias("year"),
                pl.lit(self._extraction_date.strftime("%m")).alias("month"),
                pl.lit(self._extraction_date.strftime("%d")).alias("day"),
                pl.lit(self._extraction_date.strftime("%H")).alias("hour"),
            ]

            tick_lazy = tick_lazy.with_columns(extra_exprs + partition_exprs)

            if save:
                df_collected = tick_lazy.collect()
                df_collected.columns = [clean_column_name(col) for col in df_collected.columns]
                self.write_result(df_collected, row.get("file_name", "unknown_file"))
            else:
                all_lazyframes.append(tick_lazy)

        if not save:
            combined = pl.concat(all_lazyframes).collect()
            return combined
        else:
            return None

    def write_result(self, df: pl.DataFrame, file_name: str) -> None:
        output_path = self.data_path / f"hour={self.hour}"
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / (file_name.replace(".csv", "").replace(" ", "_") + ".snappy.parquet")
        df.write_parquet(file_path, compression="snappy", statistics=True)
        self.logger.debug(f"Saved {file_path}")
        
    def save_sensor_list(self, sensors: List[Dict[str, Any]]) -> None:
        output_path = (
            Path(self.data_root)
            / "raw_data"
            / "sensor_list"
            / f"year={self.year}"
            / f"month={self.month}"
            / f"day={self.day}"
            / f"hour={self.hour}"
        )
        output_path.mkdir(parents=True, exist_ok=True)

        for chamber in sensors:
            chamber["sensor_list"] = [clean_column_name(col) for col in chamber.get("sensor_list", [])]

        sensor_df = pl.DataFrame(sensors)
        sensor_file = output_path / "sensors.snappy.parquet"
        sensor_df.write_parquet(sensor_file, compression="snappy")
        self.logger.debug(f"Saved sensor list to {sensor_file}")

    @staticmethod
    def clean_string(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9\s/]", "", s)
        s = s.replace(" ", "_")
        return s.lower()

    
    #@DuckdbBase.success_output
    @profile
    def run(self) -> str:
        self.logger.info(f"{self.custom_task_id} started")
        chamber_df, chamber_sensors = self.load_chamber_and_sensors()
        chamber_list = chamber_df["Tool/Chamber"].to_list()
        self.logger.info(f"Retrieved chamber list: {chamber_list}")

        wafer_list = self.wafer_list(chamber_list).join(chamber_df, on="Tool/Chamber", how="left")
        wafer_list.columns = [self.clean_string(c) for c in wafer_list.columns]

        wafer_list = (
            wafer_list.with_columns(
                pl.col("start_time").dt.strftime("%Y%m%d%H%M%S").alias("start_time_str"),
            )
            .with_columns(
                pl.format(
                    "tickdata_{}_{}_{}_{}_{}.csv",
                    pl.col("start_time_str"),
                    pl.col("pjid").str.replace_all("/", "_"),
                    pl.col("wafer_id"),
                    pl.col("load_port"),
                    pl.col("slot"),
                ).alias("file_name")
            )
            .drop("start_time_str")
        )
        self.wafer_list = wafer_list

        jobids, chamber_sensors_dict = self.export_tickdata_for_chambers(chamber_sensors, wafer_list)
        if not jobids:
            self.logger.warning("No jobids to process, exiting")
            return ""
        
        # pdb.set_trace()

        tick_data_df = self.download_and_parse_tickdata(jobids)
        self.save_sensor_list(chamber_sensors_dict)
        self.tick_data_df = tick_data_df

        if tick_data_df.is_empty():
            self.logger.warning("No tick data parsed, exiting")
            return ""

        self.process_and_save_tickdata(wafer_list, tick_data_df, save=True)
        del tick_data_df, self.tick_data_df
        gc.collect()
        self.logger.info(f"{self.custom_task_id} finished successfully")
        return ""


class TactrasOESLoader(TactrasLoader):
    def get_sensor_list_filter(self, chamber_sensors: pl.DataFrame, chamber_id: str) -> List[str]:
        filtered = (
            chamber_sensors.filter(
                (pl.col("Tool/Chamber") == chamber_id) & (pl.col("label")=="OES")
            )
            .explode("sensors")
            .with_columns(
                pl.col("sensors").str.split("/").list.get(0).alias("sensor"),
                pl.col("sensors").str.split("/").list.get(1).alias("isHSM"),
            )
            .filter(pl.col("isHSM") == "False")
            .filter(
                (~pl.col("sensor").str.contains("(10ms)")) & (~pl.col("sensor").str.contains("@")) & (~pl.col("sensor").str.contains("setting data"))
            )
            .select("sensor")
        )
        # sensors = filtered.to_series().to_list()
        sensors = filtered.unique(maintain_order=True).to_series().to_list()
        self.logger.info(f"Filtered sensors for chamber {chamber_id}: {len(sensors)} sensors")

        return sensors


class TickDataLoader(DuckdbBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = Path(self.data_root) / "raw_data" / "tick_data" / f"year={self.year}" / f"month={self.month}" / f"day={self.day}"

    def requires(self):
        return KnimeLoader(extraction_date=self.extraction_date)


    @DuckdbBase.success_output
    def run(self) -> str:
        parquet_path = self.data_path / f"hour={self.hour}"
        self.logger.info(f"{self.custom_task_id} is started")

        if not parquet_path.exists():
            self.logger.info(f"Data path {parquet_path} is not exists; skipping {self.custom_task_id}")
            raise DataNotExists
        parquet_path = self.data_path / f"hour={self.hour}"
        tick_datas = pl.scan_parquet(
            str(parquet_path / "*.parquet"),
            use_statistics=True,
            hive_partitioning=True,
            low_memory=True,
            allow_missing_columns=True,
            missing_columns="insert",
            extra_columns="ignore"
        ).with_columns(
            pl.col("Tool/Chamber").str.split("/").list.get(0).alias("equipment_name"),
            pl.col("Tool/Chamber").str.split("/").list.get(1).alias("chamber_name"),
        )

        for tool_chamber in pl.Series(tick_datas.select("Tool/Chamber").collect().unique()).to_list():
            #TODO: polars가 scan_parquet으로 여러 parquet 읽어들이면서 파일마다 스키마다 다른 경우 에러 발생. Cast도 불가함. 그래서 duckdb의 union_by_name=TRUE 기능으로 임시 대체. tick_data = conn.execute().pl()
            #tick_data = tick_datas.filter(pl.col("tool/chamber") == tool_chamber).collect()
            try:
                temp_con = duckdb.connect()
                tick_data = temp_con.execute(f"""
                SELECT 
                    *,
                    SPLIT_PART("Tool/Chamber", '/', 1) AS equipment_name,
                    SPLIT_PART("Tool/Chamber", '/', 2) AS chamber_name,
                FROM read_parquet('{str(parquet_path / "*.parquet")}', union_by_name=True)
                WHERE "Tool/Chamber"='{tool_chamber}'
                """).pl()
                new_columns = [clean_column_name(col) for col in tick_data.columns]
                tick_data = tick_data.rename({old: new for old, new in zip(tick_data.columns, new_columns)})
            except Exception as e:
                print(e)

            equipment_name = tick_data["equipment_name"][0]
            chamber_name = tick_data["chamber_name"][0]
            self.logger.info(f"Load from raw data finish - {equipment_name}/{chamber_name}")

            # Insert equipment
            self.conn.sql(f"INSERT INTO equipment (name)  VALUES('{equipment_name}') ON CONFLICT DO NOTHING")

            # Insert chamber
            self.conn.sql(
                f"INSERT INTO chamber (name, equipment_name) VALUES('{chamber_name}', '{equipment_name}') ON CONFLICT DO NOTHING"
            )

            # Update chamber sensor_list
            sensor_parquet_path = Path(self.data_root)  / "raw_data" / "sensor_list" / f"year={self.year}" / f"month={self.month}" / f"day={self.day}"  / f"hour={self.hour}"
            sensor_list_data = pl.scan_parquet(
                str(sensor_parquet_path / "*.parquet")
            )

            sensor_list_data = sensor_list_data.unique(subset=["chamber_id", "sensor_list"]).collect()
            sensor_lists = sensor_list_data['sensor_list'].to_list()
            cleaned_lists = [
                [clean_column_name(x) if x is not None else None for x in lst]
                if lst is not None else None
                for lst in sensor_lists
            ]
            sensor_list_df = sensor_list_data.with_columns(pl.Series("sensor_list", cleaned_lists))
            self.conn.sql(
                rf"""
                UPDATE chamber AS c
                SET sensor_list = p.sensor_list
                FROM (
                    SELECT
                        chamber_id,
                        SPLIT_PART(chamber_id, '/', 1) AS equipment_name,
                        SPLIT_PART(chamber_id, '/', 2) AS name,
                        sensor_list
                    FROM sensor_list_df
                    WHERE "chamber_id"='{equipment_name}/{chamber_name}'
                ) AS p
                WHERE c.equipment_name = p.equipment_name
                AND c.name = p.name;
                """, obj_list={"sensor_list_df":sensor_list_df, 'tick_data':tick_data}
            )

            # Insert recipe
            self.conn.sql(
                """
                WITH recipe_ref AS (
                    SELECT 
                        equipment_name,
                        chamber_name,
                        system_recipe,
                        process_recipe,
                        ARRAY_AGG(step ORDER BY step) AS step_list
                    FROM (
                        SELECT DISTINCT equipment_name, chamber_name, system_recipe, process_recipe, step
                        FROM tick_data
                    )
                    GROUP BY equipment_name, chamber_name, system_recipe, process_recipe
                )
                INSERT INTO recipe (system_recipe, process_recipe, step_list)
                SELECT DISTINCT system_recipe, process_recipe, step_list
                FROM recipe_ref
                ON CONFLICT DO NOTHING;
                """
            )

            # Insert wafer_operation_log
            self.conn.sql(
                """
                INSERT INTO wafer_operation_log (
                    jid, equipment_id, equipment_name,
                    chamber_id, chamber_name, recipe_id,
                    system_recipe, process_recipe, cjid,
                    pjid, carrier_id, wafer_id,
                    wafer_type, recipe_type, recipe_status, data_type,
                    start_time, end_time, created_at
                )
                SELECT
                    td.jid, e.id AS equipment_id, td.equipment_name,
                    c.id AS chamber_id, td.chamber_name, r.id AS recipe_id,
                    td.system_recipe, td.process_recipe, td.cjid,
                    td.pjid, td.carrier_id, td.wafer_id, td.wafer_type, td.recipe_type,
                    td.recipe_status, td.data_type, td.start_time, td.end_time,
                    current_localtimestamp() AS created_at
                FROM (
                    SELECT DISTINCT
                        jid, equipment_name, chamber_name,
                        system_recipe, process_recipe,
                        cjid, pjid, carrier_id,
                        wafer_id, wafer_type, recipe_type,
                        recipe_status, contents AS data_type,
                        start_time, end_time
                    FROM tick_data
                ) td
                JOIN equipment e ON td.equipment_name = e.name
                JOIN chamber c ON td.chamber_name = c.name AND td.equipment_name = c.equipment_name
                JOIN recipe r ON td.system_recipe = r.system_recipe AND td.process_recipe = r.process_recipe
                ON CONFLICT DO NOTHING;
                """,
            )

            # Insert equipment_structure
            self.conn.sql(
                """
                INSERT INTO equipment_structure (
                    equipment_id,
                    chamber_id,
                    recipe_id,
                    equipment_name,
                    equipment_type,
                    chamber_name,
                    chamber_type,
                    system_recipe,
                    process_recipe,
                    recipe_alias
                )
                SELECT
                    wl.equipment_id,
                    wl.chamber_id,
                    wl.recipe_id,
                    wl.equipment_name,
                    e.type AS equipment_type,
                    wl.chamber_name,
                    c.type AS chamber_type,
                    r.system_recipe,
                    r.process_recipe,
                    r.alias AS recipe_alias
                FROM (
                    SELECT DISTINCT
                        equipment_id, chamber_id, recipe_id,
                        equipment_name, chamber_name,
                        system_recipe, process_recipe
                    FROM wafer_operation_log
                ) wl
                JOIN equipment e ON wl.equipment_id = e.id
                JOIN chamber c ON wl.chamber_id = c.id
                JOIN recipe r ON wl.recipe_id = r.id
                ON CONFLICT DO NOTHING;
                """
            )

            self.conn.sql(
                """
                INSERT INTO equipment_type_structure (
                    equipment_type, chamber_type, recipe_alias, name, sensor_list, step_list, updated_at
                )
                SELECT
                    equipment_type,
                    chamber_type,
                    recipe_alias,
                    concat(equipment_type, '-', chamber_type, '-', recipe_alias) AS name,
                    list_sort(list_distinct(flatten(list(sensor_list)))) AS sensor_list,
                    list_sort(list_distinct(flatten(list(step_list)))) AS step_list,
                    current_localtimestamp() AS updated_at
                FROM (
                    SELECT es.*, c.sensor_list, r.step_list
                    FROM equipment_structure es
                    JOIN chamber c ON es.chamber_id = c.id
                    JOIN recipe r ON es.recipe_id = r.id
                ) sub
                GROUP BY equipment_type, chamber_type, recipe_alias
                HAVING equipment_type IS NOT NULL AND chamber_type IS NOT NULL AND recipe_alias IS NOT NULL
                ON CONFLICT (equipment_type, chamber_type, recipe_alias)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    sensor_list = EXCLUDED.sensor_list,
                    step_list = EXCLUDED.step_list,
                    updated_at = EXCLUDED.updated_at,
                    id = equipment_type_structure.id;
                """
            )

            # Create temporary ets_table
            self.conn.sql(
                """
                CREATE TEMP TABLE IF NOT EXISTS ets_table AS (
                    SELECT ets.name AS equipment_type_structure_name, es.*, ets.sensor_list
                    FROM equipment_structure AS es
                    JOIN equipment_type_structure AS ets
                        ON es.equipment_type = ets.equipment_type
                        AND es.chamber_type = ets.chamber_type
                        AND es.recipe_alias = ets.recipe_alias
                );
                """
            )

            base_query = """
                SELECT * FROM ets_table es
                JOIN tick_data td ON es.equipment_name = td.equipment_name
                    AND es.chamber_name = td.chamber_name
                    AND es.process_recipe = td.process_recipe
                    AND es.system_recipe = td.system_recipe
            """
            # data = self.conn.sql(base_query)

            sensors = (
                self.conn.sql(
                    f"""
                    SELECT equipment_type_structure_name, 
                        list_sort(list_distinct(flatten(list(sensor_list)))) AS sensor_list
                    FROM ({base_query})
                    GROUP BY equipment_type_structure_name
                    """
                )
                .to_df()
                .to_dict(orient="records")
            )

            meta_cols = [
                "jid",
                "wafer_id",
                "es.equipment_name",
                "es.chamber_name",
                "es.system_recipe",
                "es.process_recipe",
                "es.equipment_type_structure_name",
                "time",
                "step",
                "sequence",
                "year",
                "month",
                "day",
                "hour"
            ]

            for sensor_record in sensors:
                tmp_tick_data = tick_data
                tmp_tick_data.columns = [clean_column_name(col) for col in tmp_tick_data.columns ]
                sensor_record["sensor_list"] = [clean_column_name(col) for col in sensor_record["sensor_list"]]

                self.logger.info(sensor_record)
                tmp_tick_data = tmp_tick_data.with_columns([
                    pl.lit(None).alias(col) for col in list(set(sensor_record['sensor_list']) - set(tmp_tick_data.columns))
                ])

                select_sensor = ', '.join([f"\"{remove_sl}\"" if "." not in remove_sl else remove_sl for remove_sl in sensor_record["sensor_list"]])
                select_sensor = ', '.join(meta_cols) + ', ' + select_sensor

                filtered_query = (
                    base_query.replace("*", select_sensor)
                    + f" WHERE es.equipment_type_structure_name = '{sensor_record['equipment_type_structure_name']}'"
                )

                dst_path = (
                    Path(self.data_root)
                    / "cleaned_data"
                    / sensor_record["equipment_type_structure_name"]
                    / f"year={self._extraction_date.year}"
                    / f"month={self._extraction_date.month}"
                    / f"day={self._extraction_date.day}"
                    / f"hour={self._extraction_date.hour}"
                )

                new_query = self._save_parquet_without_partitioning(filtered_query, str(dst_path), over_write=False)
                self.logger.info(f"[SQL] {new_query}")
                self.conn.sql(new_query, obj_list={'tick_data': tmp_tick_data})

        return "TickDataLoader Is Finished"

