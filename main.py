import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta


def parse_log(filepath):
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(INFO|ERROR|WARN)\s*\|\s*"
        r"service=(auth|payments|notifications)\s*\|\s*latency_ms=(\d+)\s*\|\s*status=(\w+)"
    )
    data = []
    broken_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                data.append(match.groups())
            else:
                broken_count += 1

    df = pd.DataFrame(
        data, columns=["timestamp", "level", "service", "latency_ms", "status"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["latency_ms"] = pd.to_numeric(df["latency_ms"])

    print(f"--- Базовая статистика парсинга ---")
    print(f"Успешно распарсено строк: {len(df)}")
    print(f"Пропущено битых строк: {broken_count}\n")

    return df


def print_basic_stats(df):
    print("--- Распределение по сервисам ---")
    print(df["service"].value_counts())
    print("\n--- Распределение статусов ---")
    print(df["status"].value_counts())

    errors = df[df["status"].isin(["error", "timeout"])]
    error_rate = (len(errors) / len(df)) * 100
    print(f"\nОбщий процент ошибок (error + timeout): {error_rate:.2f}%\n")


def detect_anomalies(df, window="1h", n_std=3.0, incident_gap="5min"):
    df = df.sort_values(["service", "timestamp"]).copy()
    incidents = []

    for service, group in df.groupby("service"):
        group = group.set_index("timestamp").sort_index()

        roll_mean = group["latency_ms"].rolling(window, min_periods=1).mean()
        roll_std = group["latency_ms"].rolling(window, min_periods=20).std()

        # Считаем выбросами значения latency,
        # которые превышают локальное среднее более чем на 3 сигмы
        threshold = roll_mean + n_std * roll_std
        group["is_anomaly"] = group["latency_ms"] > threshold

        anomalous_points = group[group["is_anomaly"]]
        if anomalous_points.empty:
            continue

        time_diffs = anomalous_points.index.to_series().diff()
        incident_ids = (time_diffs > pd.Timedelta(incident_gap)).cumsum()
        for inc_id, inc_group in anomalous_points.groupby(incident_ids):
            start_t = inc_group.index.min()
            end_t = inc_group.index.max()

            incidents.append(
                {
                    "service": service,
                    "start_time": start_t,
                    "end_time": end_t,
                    "duration_min": (end_t - start_t).total_seconds() / 60,
                    "max_latency": inc_group["latency_ms"].max(),
                    "anomaly_count": len(inc_group),
                }
            )

    incidents_df = pd.DataFrame(incidents)

    print("--- Найденные инциденты ---")
    if not incidents_df.empty:
        print(incidents_df.to_string(index=False))
    else:
        print("Инцидентов не найдено.")
    print("-" * 30)

    return incidents_df


def plot_results(df, incidents_df):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 15),
        sharex=False,
    )
    services = df["service"].unique()

    for i, service in enumerate(services):
        ax = axes[i]
        sub_df = df[df["service"] == service].sort_values("timestamp")

        ax.plot(
            sub_df["timestamp"],
            sub_df["latency_ms"],
            color="blue",
            alpha=0.2,
            linewidth=0.5,
            label="Обычные запросы",
        )

        if not incidents_df.empty:
            service_incidents = incidents_df[incidents_df["service"] == service]
            for _, inc in service_incidents.iterrows():
                pad = pd.Timedelta("1min")
                ax.axvspan(
                    inc["start_time"] - pad,
                    inc["end_time"] + pad,
                    color="red",
                    alpha=0.3,
                    label="Инцидент" if i == 0 else "",
                )

                anom_mask = (sub_df["timestamp"] >= inc["start_time"]) & (
                    sub_df["timestamp"] <= inc["end_time"]
                )
                ax.scatter(
                    sub_df[anom_mask]["timestamp"],
                    sub_df[anom_mask]["latency_ms"],
                    color="darkred",
                    s=15,
                    zorder=5,
                )

        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.set_title(f"Сервис: {service.upper()}", fontweight="bold", fontsize=12)

        if i < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Время", fontsize=10)

    plt.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.95,
        bottom=0.05,
        hspace=0.25,
    )

    plt.savefig("anomalies_visualization.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    LOG_FILE = "service_log.txt"

    # Парсинг
    df = parse_log(LOG_FILE)
    print_basic_stats(df)

    # Поиск аномалий
    incidents_df = detect_anomalies(df, window="1h", n_std=3.0, incident_gap="5min")

    # Визуализация
    plot_results(df, incidents_df)
