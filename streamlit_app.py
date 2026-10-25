"""
판넬 빌더 오더 납기 컨트롤 타워 (Streamlit 시작 코드)

실행: streamlit run app.py
데이터: data/orders.csv, data/eta_history.csv (generate_data.py 로 생성, 사이드바에서 새 오더 파일로 업데이트)
"""

import re
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
AS_OF_FILE = DATA_DIR / "as_of.txt"


def read_as_of() -> pd.Timestamp:
    """기준일: 마지막으로 반영한 오더 파일의 날짜 (파일이 없으면 가상 데이터 기준일)"""
    try:
        return pd.Timestamp(AS_OF_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return pd.Timestamp("2026-10-23")  # generate_data.py 의 기준일과 동일


TODAY = read_as_of()
OPEN_STATUSES = ["발주 완료", "생산 중", "운송 중"]
RISK_WINDOW_DAYS = 30
AIR_DAYS = 10

# 색상: 단일 계열은 파랑, 비교 기준(계획)은 회색, 심각도는 상태 색상
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#a3a29c"
SEVERITY_COLORS = {"심각": "#d03b3b", "주의": "#ec835a", "경미": "#fab219"}

st.set_page_config(page_title="PB 오더 납기 컨트롤 타워", page_icon="📦", layout="wide")


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------
ORDER_DATE_COLUMNS = [
    "order_date", "CRD", "CDD", "planned_ship_date", "revised_ship_date",
    "original_ETA", "revised_ETA", "actual_delivery_date",
]


@st.cache_data
def load_data(as_of: pd.Timestamp):
    # as_of 는 캐시 구분용: 새 파일을 반영해 기준일이 바뀌면 다시 읽음
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=ORDER_DATE_COLUMNS, encoding="utf-8-sig", dtype={"order_no": str})
    history = pd.read_csv(DATA_DIR / "eta_history.csv", parse_dates=["snapshot_date", "old_ETA", "new_ETA"],
                          encoding="utf-8-sig", dtype={"order_no": str})

    # 지연 여부는 업로드 파일에 없어도 되도록 날짜로 다시 계산
    cancelled = orders["status"] == "취소"
    final_eta = orders["actual_delivery_date"].fillna(orders["revised_ETA"])
    orders["delay_days_vs_CRD"] = (final_eta - orders["CRD"]).dt.days.clip(lower=0).where(~cancelled, 0).astype(int)
    orders["is_delayed"] = ~cancelled & (orders["delay_days_vs_CRD"] > 0)
    orders["amount"] = orders["quantity"] * orders["unit_price"]
    orders["expedited"] = orders["original_shipping_mode"].eq("SEA") & orders["shipping_mode"].eq("AIR")

    orders["line_key"] = orders["order_no"] + "-" + orders["line_no"].astype(str)
    orders["open_qty"] = orders["quantity"] - orders["shipped_qty"]
    orders["open_amount"] = orders["open_qty"] * orders["unit_price"]
    orders["is_open"] = orders["status"].isin(OPEN_STATUSES)
    orders["is_partial"] = orders["shipped_qty"].between(1, orders["quantity"] - 1)
    # 공장 지연이 없어도 최초 확정일 자체가 CRD 보다 늦으면 지연으로 잡히므로 사유를 채워줌
    no_reason = orders["is_delayed"] & orders["delay_reason"].fillna("").eq("")
    orders["delay_reason"] = orders["delay_reason"].fillna("")
    orders.loc[no_reason, "delay_reason"] = "최초 확정일이 CRD 이후"
    orders["customer_reason"] = orders["delay_reason"].map(CUSTOMER_REASON).fillna("")

    change_count = history.groupby(["order_no", "line_no"]).size().rename("eta_change_count")
    orders = orders.merge(change_count, on=["order_no", "line_no"], how="left")
    orders["eta_change_count"] = orders["eta_change_count"].fillna(0).astype(int)
    return orders, history, build_milestones(orders)


def won_eok(value: float) -> str:
    return f"{value / 1e8:,.1f}억 원"


def pct(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "-"


# ---------------------------------------------------------------------------
# KPI 계산
# ---------------------------------------------------------------------------
def calc_kpis(df: pd.DataFrame) -> dict:
    open_df = df[df["is_open"]]
    delivered = df[df["status"] == "입고 완료"]
    delayed = df[df["is_delayed"]]
    risk = risk_lines(df)
    return {
        "backlog": open_df["open_amount"].sum(),
        "open_lines": len(open_df),
        "otd_crd": pct((delivered["actual_delivery_date"] <= delivered["CRD"]).sum(), len(delivered)),
        "otd_cdd": pct((delivered["actual_delivery_date"] <= delivered["CDD"]).sum(), len(delivered)),
        "delivered_lines": len(delivered),
        "avg_delay": delayed["delay_days_vs_CRD"].mean() if len(delayed) else 0,
        "delayed_lines": len(delayed),
        "partial_ratio": pct(open_df["is_partial"].sum(), len(open_df)),
        "risk_lines": len(risk),
        "risk_orders": risk["order_no"].nunique(),
    }


def risk_lines(df: pd.DataFrame) -> pd.DataFrame:
    """30일 내 위험 오더: CRD 가 앞으로 30일 안에 도래하는데 현재 ETA 가 CRD 보다 늦은 미입고 라인"""
    window_end = TODAY + pd.Timedelta(days=RISK_WINDOW_DAYS)
    return df[df["is_open"] & df["CRD"].between(TODAY, window_end) & (df["revised_ETA"] > df["CRD"])]


# ---------------------------------------------------------------------------
# 조치 제안
# ---------------------------------------------------------------------------
def suggest_action(row) -> tuple[str, int]:
    """(조치 제안, AIR 전환 시 단축 일수)"""
    not_shipped = row["revised_ship_date"] > TODAY
    if row["shipping_mode"] == "SEA" and not_shipped:
        air_eta = row["revised_ship_date"] + pd.Timedelta(days=AIR_DAYS)
        saved = (row["revised_ETA"] - air_eta).days
        if air_eta <= row["CRD"]:
            return f"AIR 전환 시 CRD 충족 ({saved}일 단축)", saved
        return f"AIR 전환 검토 ({saved}일 단축, CRD 대비 {(air_eta - row['CRD']).days}일 지연 잔존)", saved
    if row["delay_reason"] == "고객 사양 변경":
        return "변경 사양 확정 후 고객과 납기 재협의", 0
    if row["status"] == "운송 중":
        if row["is_partial"]:
            return "선입고분 우선 제작 협의, 잔량 출하 일정 확인", 0
        return "통관·입고 일정 확인 후 고객에 변경 ETA 공유", 0
    if row["shipping_mode"] == "AIR":
        return "공장에 생산 일정 단축(Expedite) 요청", 0
    return "공급망팀과 출하 일정 재확인", 0


def severity(days: int) -> str:
    """CRD 대비 지연일 기준: 30일 이상 심각, 8~29일 주의, 1~7일 경미"""
    if days >= 30:
        return "심각"
    if days >= 8:
        return "주의"
    return "경미"


def crd_bucket(crd: pd.Timestamp) -> str:
    if crd < TODAY:
        return "CRD 경과"
    if crd <= TODAY + pd.Timedelta(days=RISK_WINDOW_DAYS):
        return f"{RISK_WINDOW_DAYS}일 내 도래"
    return f"{RISK_WINDOW_DAYS}일 이후"


def build_exceptions(df: pd.DataFrame) -> pd.DataFrame:
    exc = df[df["is_open"] & df["is_delayed"]].copy()
    if exc.empty:
        return exc.assign(severity="", crd_bucket="", action="", air_saving_days=0)
    exc["severity"] = exc["delay_days_vs_CRD"].map(severity)
    exc["crd_bucket"] = exc["CRD"].map(crd_bucket)
    actions = exc.apply(suggest_action, axis=1, result_type="expand")
    exc["action"], exc["air_saving_days"] = actions[0], actions[1]
    return exc.sort_values("delay_days_vs_CRD", ascending=False)


# ---------------------------------------------------------------------------
# 지난주 대비 변화 (ETA 변경 이력으로 지난주 시점의 ETA 를 복원해서 비교)
# ---------------------------------------------------------------------------
WEEK_START = TODAY - pd.Timedelta(days=6)   # 이번 주: 기준일 포함 최근 7일
LAST_WEEK = TODAY - pd.Timedelta(days=7)    # 지난주 기준일


def eta_as_of(df: pd.DataFrame, history: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """as_of 시점까지 통보된 변경만 반영한 ETA (변경이 없으면 최초 ETA)"""
    known = history[history["snapshot_date"] <= as_of].sort_values(["snapshot_date", "change_seq"])
    last_eta = known.groupby(["order_no", "line_no"])["new_ETA"].last().rename("eta_then")
    merged = df[["order_no", "line_no", "original_ETA"]].merge(last_eta, on=["order_no", "line_no"], how="left")
    return merged["eta_then"].fillna(merged["original_ETA"]).set_axis(df.index)


def weekly_changes(df: pd.DataFrame, history: pd.DataFrame) -> dict:
    df = df[df["status"] != "취소"].copy()
    df["eta_last_week"] = eta_as_of(df, history, LAST_WEEK)
    df["shift_this_week"] = (df["revised_ETA"] - df["eta_last_week"]).dt.days

    # 지난주 시점에 미입고였던 라인과 그때의 지연 여부
    open_then = (df["order_date"] <= LAST_WEEK) & (df["actual_delivery_date"].isna() | (df["actual_delivery_date"] > LAST_WEEK))
    delayed_then = open_then & (df["eta_last_week"] > df["CRD"])
    delayed_now = df["is_open"] & df["is_delayed"]

    new_delay = df[delayed_now & ~delayed_then].copy()
    new_delay["cause"] = new_delay["order_date"].gt(LAST_WEEK).map({True: "이번 주 신규 발주", False: "ETA 변경"})
    resolved = df[delayed_then & df["is_open"] & ~df["is_delayed"]].copy()
    delivered = df[df["actual_delivery_date"].between(WEEK_START, TODAY)]

    changes = history[history["snapshot_date"].between(WEEK_START, TODAY)].merge(
        df[["order_no", "line_no", "line_key", "project_name", "sub_project_name", "product_family", "CRD"]],
        on=["order_no", "line_no"],
    )
    return {
        "delayed_now": int(delayed_now.sum()),
        "delayed_then": int(delayed_then.sum()),
        "new_delay": new_delay.sort_values("delay_days_vs_CRD", ascending=False),
        "resolved": resolved,
        "delivered": delivered,
        "delivered_late": int((delivered["actual_delivery_date"] > delivered["CRD"]).sum()),
        "changes": changes.sort_values(["snapshot_date", "line_key"], ascending=[False, True]),
    }


def weekly_change_trend(df: pd.DataFrame, history: pd.DataFrame, weeks: int = 8) -> pd.DataFrame:
    """최근 몇 주간 주별 ETA 변경 건수 (늦춰짐 / 앞당김)"""
    keys = df[["order_no", "line_no"]]
    hist = history.merge(keys, on=["order_no", "line_no"])
    hist = hist[hist["snapshot_date"] > TODAY - pd.Timedelta(days=7 * weeks)].copy()
    # 기준일에서 7일 단위로 거꾸로 묶음 (0 = 이번 주)
    hist["weeks_ago"] = (TODAY - hist["snapshot_date"]).dt.days // 7
    hist["week_end"] = (TODAY - pd.to_timedelta(hist["weeks_ago"] * 7, unit="D")).dt.strftime("~%m-%d")
    hist["direction"] = hist["shift_days"].gt(0).map({True: "늦춰짐", False: "앞당김"})
    return hist.groupby(["week_end", "direction"], as_index=False).size().sort_values("week_end")


# ---------------------------------------------------------------------------
# 보고 대상별 보기 모드
# ---------------------------------------------------------------------------
# amount: 금액 표시 / internal: 내부 지연 사유·조치 제안 표시 / line_detail: 라인 단위 표 표시 / supply: 공장·출하 중심
VIEW_MODES = {
    "전체 (내부용)": dict(amount=True, internal=True, line_detail=True, supply=False,
                       desc="모든 지표와 표를 보여줍니다."),
    "영업 관리자용": dict(amount=True, internal=True, line_detail=False, supply=False,
                       desc="KPI와 프로젝트 요약 중심으로 보여주고, 라인 단위 상세 표는 숨깁니다."),
    "고객 공유용": dict(amount=False, internal=False, line_detail=True, supply=False,
                      desc="금액·단가, 공장별 내부 지연 사유, 조치 제안을 숨기고 오더별 변경 납기만 보여줍니다."),
    "공급망팀용": dict(amount=False, internal=True, line_detail=True, supply=True,
                     desc="공장·운송 방식별 지연과 AIR 전환 후보 중심으로 보여줍니다."),
}

# 고객에게는 내부 사정 대신 일반화한 사유로 안내
CUSTOMER_REASON = {
    "공장 생산 캐파 부족": "제조 일정 조정",
    "부품 수급 지연": "제조 일정 조정",
    "선적 스케줄 변경": "물류 일정 변경",
    "통관 지연": "물류 일정 변경",
    "고객 사양 변경": "사양 변경 반영",
    "최초 확정일이 CRD 이후": "최초 회신 납기가 요청일 이후",
    "AIR 전환 (긴급 운송)": "항공 운송 전환으로 납기 단축",
    "공장 일정 앞당김": "제조 일정 단축",
    "사유 미기재": "일정 변경",
}

# 화면 표·엑셀 공통 컬럼 (원본 컬럼 → 표시 이름)
FULL_COLUMNS = {
    "severity": "심각도", "crd_bucket": "CRD 도래", "order_no": "오더 번호", "line_no": "라인",
    "project_name": "프로젝트", "sub_project_name": "서브 프로젝트", "product_family": "제품군", "product_code": "제품 코드",
    "source_plant": "공장", "shipping_mode": "운송", "quantity": "수량", "shipped_qty": "출하 수량", "status": "상태",
    "CRD": "CRD", "CDD": "최초 확정일(CDD)", "revised_ETA": "현재 ETA", "delay_days_vs_CRD": "CRD 대비 지연일",
    "eta_change_count": "ETA 변경 횟수", "delay_reason": "지연 사유", "action": "조치 제안",
}
CUSTOMER_COLUMNS = {
    "order_no": "오더 번호", "line_no": "라인", "project_name": "프로젝트", "sub_project_name": "서브 프로젝트",
    "product_family": "제품군", "product_code": "제품 코드", "quantity": "수량", "shipped_qty": "출하 수량",
    "CRD": "요청 납기(CRD)", "revised_ETA": "변경 납기 예정일", "delay_days_vs_CRD": "요청 대비 지연일",
    "eta_change_count": "납기 변경 횟수", "customer_reason": "변경 사유",
}
SUPPLY_COLUMNS = {
    "severity": "심각도", "order_no": "오더 번호", "line_no": "라인", "product_family": "제품군", "product_code": "제품 코드",
    "source_plant": "공장", "shipping_mode": "운송", "status": "상태", "revised_ship_date": "출하 예정일",
    "revised_ETA": "현재 ETA", "CRD": "CRD", "delay_days_vs_CRD": "CRD 대비 지연일", "delay_reason": "지연 사유",
    "air_saving_days": "AIR 전환 시 단축일", "action": "조치 제안",
}
MANAGER_TOP_COLUMNS = {
    "order_no": "오더 번호", "line_no": "라인", "project_name": "프로젝트", "product_family": "제품군",
    "CRD": "CRD", "revised_ETA": "현재 ETA", "delay_days_vs_CRD": "CRD 대비 지연일", "open_amount": "잔량 금액(원)",
    "action": "조치 제안",
}


def table(frame: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """원본 컬럼을 골라 표시 이름으로 바꾸고, 날짜는 시간 없이 날짜만 남김"""
    out = frame[list(columns)].rename(columns=columns)
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.date
    return out


def project_summary(exceptions: pd.DataFrame) -> pd.DataFrame:
    counts = exceptions.pivot_table(index="project_name", columns="severity", values="line_key", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=list(SEVERITY_COLORS), fill_value=0)
    stats = exceptions.groupby("project_name").agg(
        지연_라인=("line_key", "count"), 최대_지연일=("delay_days_vs_CRD", "max"), 관련_backlog=("open_amount", "sum"),
    )
    out = stats.join(counts).sort_values("관련_backlog", ascending=False).reset_index()
    out["관련_backlog"] = out["관련_backlog"].map(won_eok)
    return out.rename(columns=lambda c: {"project_name": "프로젝트"}.get(c, c.replace("_", " ")))


def history_table(exceptions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    keys = exceptions[["order_no", "line_no"]]
    hist = history.merge(keys, on=["order_no", "line_no"]).sort_values(["order_no", "line_no", "change_seq"])
    return table(hist, {
        "snapshot_date": "변경 통보일", "order_no": "오더 번호", "line_no": "라인", "change_seq": "변경 차수",
        "old_ETA": "이전 ETA", "new_ETA": "변경 ETA", "shift_days": "변동 일수", "reason": "사유", "shipping_mode": "운송",
    })


# ---------------------------------------------------------------------------
# 엑셀 출력 (보기 모드별로 시트 구성이 달라짐)
# ---------------------------------------------------------------------------
def build_report(mode_name: str, view: pd.DataFrame, history: pd.DataFrame) -> tuple[dict, str]:
    if mode_name == "영업 관리자용":
        top = view[view["severity"] == "심각"].head(20)
        sheets = {"프로젝트 요약": project_summary(view), "심각 지연 TOP 20": table(top, MANAGER_TOP_COLUMNS)}
        return sheets, "PB_manager_summary"
    if mode_name == "고객 공유용":
        return {"납기 변경 안내": table(view, CUSTOMER_COLUMNS)}, "PB_delivery_update"
    if mode_name == "공급망팀용":
        return {"지연 라인 (공급망)": table(view, SUPPLY_COLUMNS), "ETA 변경 이력": history_table(view, history)}, "PB_supply_followup"
    sheets = {
        "지연·위험 오더": table(view, FULL_COLUMNS),
        "프로젝트 요약": project_summary(view),
        "ETA 변경 이력": history_table(view, history),
    }
    return sheets, "PB_delay_report"


def to_excel(sheets: dict) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(width * 1.6 + 2, 60)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 메일 초안 (Outlook 에서 열 수 있는 .eml 초안 + 엑셀 첨부)
# ---------------------------------------------------------------------------
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

CUSTOMER_SCHEDULE_COLUMNS = {
    "order_no": "오더 번호", "line_no": "라인", "sub_project_name": "서브 프로젝트", "product_family": "제품군",
    "product_code": "제품 코드", "quantity": "수량", "shipped_qty": "출하 수량", "status": "진행 상태",
    "CRD": "요청 납기(CRD)", "revised_ETA": "납기 예정일", "delay_days_vs_CRD": "요청 대비 지연일", "customer_reason": "변경 사유",
}
SUPPLY_REQUEST_COLUMNS = {
    "order_no": "오더 번호", "line_no": "라인", "project_name": "프로젝트", "product_family": "제품군",
    "product_code": "제품 코드", "source_plant": "공장", "shipping_mode": "운송", "quantity": "수량",
    "shipped_qty": "출하 수량", "status": "상태", "revised_ship_date": "출하 예정일", "revised_ETA": "현재 ETA",
    "CRD": "고객 요청 납기(CRD)", "delay_days_vs_CRD": "필요 단축일", "delay_reason": "지연 사유", "request": "요청 사항",
}


def supply_request(row) -> str:
    """공급망 담당자에게 요청할 내용"""
    if row["air_saving_days"] > 0:
        return f"AIR 전환 가능 여부 및 비용 확인 (전환 시 약 {row['air_saving_days']}일 단축)"
    if row["status"] == "운송 중":
        return "통관·입고 일정 단축 가능 여부 확인"
    return "생산·출하 일정 단축(Expedite) 가능 여부 확인"


def build_eml(to: str, subject: str, body: str, attachment: bytes, filename: str) -> bytes:
    """X-Unsent 헤더가 있는 .eml 은 Outlook 에서 열면 보내기 전 초안 상태로 열림"""
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg["X-Unsent"] = "1"
    msg.set_content(body)
    maintype, subtype = XLSX_MIME.split("/")
    msg.add_attachment(attachment, maintype=maintype, subtype=subtype, filename=filename)
    return msg.as_bytes()


def customer_mail(project: str, sender: str) -> tuple[str, str]:
    subject = f"[{project}] 납기 일정 안내_{TODAY:%Y%m%d}"
    body = (
        "담당자님.\n"
        f"안녕하세요, 슈나이더일렉트릭코리아 {sender}입니다.\n"
        "표제의 프로젝트 건 납기일정 전달 드립니다.\n"
        "첨부된 파일 참고하시기 바랍니다.\n\n"
        "관련된 문의 사항이 있으실 경우, 언제든지 연락해 주시기 바랍니다.\n\n"
        f"{sender} 드림"
    )
    return subject, body


def supply_mail(order_nos: list[str], manager: str, sender: str) -> tuple[str, str]:
    orders_text = order_nos[0] if len(order_nos) == 1 else f"{order_nos[0]} 외 {len(order_nos) - 1}건"
    subject = f"[납기 확인 및 단축 요청] {orders_text}_{TODAY:%Y%m%d}"
    body = (
        f"{manager} 매니저님.\n"
        f"안녕하세요, 영업팀 {sender} 입니다.\n"
        "표제의 오더 건 업데이트 된 납기 일정 및 납기 단축 요청드립니다.\n"
        "자세한 사항은 첨부된 파일 참고하시기 바랍니다.\n\n"
        f"{sender} 드림"
    )
    return subject, body


def mail_actions(to: str, subject: str, body: str, attachment: bytes, filename: str, key: str):
    """Outlook 초안(.eml) 다운로드, 메일 앱 바로 열기(첨부 제외), 첨부 엑셀 다운로드"""
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "📧 Outlook 초안 다운로드 (.eml)", data=build_eml(to, subject, body, attachment, filename),
        file_name=f"{Path(filename).stem}.eml", mime="message/rfc822", type="primary", width="stretch", key=f"{key}_eml",
    )
    mailto = f"mailto:{quote(to)}?subject={quote(subject)}&body={quote(body)}"
    c2.link_button("메일 앱에서 바로 열기 (첨부 제외)", mailto, width="stretch")
    c3.download_button("📎 첨부 엑셀만 다운로드", data=attachment, file_name=filename, mime=XLSX_MIME, width="stretch", key=f"{key}_xlsx")
    st.caption(".eml 파일을 더블클릭하면 Outlook 에서 엑셀이 첨부된 초안으로 열립니다. 내용 확인 후 보내기만 누르면 됩니다.")


# ---------------------------------------------------------------------------
# AIR 전환 시뮬레이션
# ---------------------------------------------------------------------------
FAT_DAYS, SITE_DAYS = 5, 20  # generate_data.py 의 단계 기간과 동일


def phase_schedule(order_start, order_end, material_end, fab_days: int, plan_type: str) -> pd.DataFrame:
    """자재 입고 완료일을 기준으로 판넬 제작 → FAT → 현장 설치 일정을 다시 계산"""
    fab_start = material_end + pd.Timedelta(days=1)
    fab_end = fab_start + pd.Timedelta(days=fab_days)
    fat_end = fab_end + pd.Timedelta(days=FAT_DAYS)
    site_end = fat_end + pd.Timedelta(days=SITE_DAYS)
    return pd.DataFrame({
        "phase": ["1. 발주", "2. 자재 공급 (슈나이더)", "3. 판넬 제작", "4. FAT", "5. 현장 납품·설치"],
        "start": [order_start, order_start, fab_start, fab_end + pd.Timedelta(days=1), fat_end + pd.Timedelta(days=1)],
        "end": [order_end, material_end, fab_end, fat_end, site_end],
        "plan_type": plan_type,
    })


def build_milestones(orders: pd.DataFrame) -> pd.DataFrame:
    """서브 프로젝트별 단계 일정 (고객 계획: 모든 자재가 CRD 에 입고 / 현재 예상: 실제 입고일·현재 ETA 기준)"""
    active = orders[orders["status"] != "취소"].copy()
    active["material_date"] = active["actual_delivery_date"].fillna(active["revised_ETA"])
    frames = []
    for (project, sub_project), g in active.groupby(["project_name", "sub_project_name"]):
        fab_days = 45 if "BlokSeT" in sub_project else 30
        order_start, order_end = g["order_date"].min(), g["order_date"].max()
        gating = g.loc[g["material_date"].idxmax()]
        for plan_type, material_end in [("고객 계획", g["CRD"].max()), ("현재 예상", g["material_date"].max())]:
            ms = phase_schedule(order_start, order_end, material_end, fab_days, plan_type)
            ms["gating_line"] = ""
            if plan_type == "현재 예상":
                ms.loc[ms["phase"].str.startswith("2."), "gating_line"] = f"{gating['line_key']} {gating['product_family']}"
            frames.append(ms.assign(project_name=project, sub_project_name=sub_project))
    return pd.concat(frames, ignore_index=True)


def air_candidates(lines: pd.DataFrame) -> pd.DataFrame:
    """아직 출하 전인 SEA 라인만 AIR 로 바꿀 수 있음"""
    cand = lines[lines["is_open"] & (lines["shipping_mode"] == "SEA") & (lines["revised_ship_date"] > TODAY)].copy()
    cand["air_eta"] = cand["revised_ship_date"] + pd.Timedelta(days=AIR_DAYS)
    return cand


def minimal_air_set(lines: pd.DataFrame, cand: pd.DataFrame, target=None) -> list:
    """병목 라인을 하나씩 AIR 로 바꿔서, 자재 입고 완료가 target 이내가 되는(없으면 최대한 당기는) 최소 조합"""
    eta = lines["material_date"].copy()
    chosen = []
    while True:
        top = eta.idxmax()
        if (target is not None and eta[top] <= target) or top not in cand.index or top in chosen:
            return chosen
        chosen.append(top)
        eta[top] = cand.at[top, "air_eta"]


# ---------------------------------------------------------------------------
# 오더 파일 업데이트 (새 파일을 어제 파일과 비교해 ETA 변경 이력을 쌓음)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "order_no", "line_no", "order_date", "customer_code", "project_name", "sub_project_name", "product_family",
    "product_code", "source_plant", "quantity", "shipped_qty", "unit_price", "original_shipping_mode", "shipping_mode",
    "CRD", "CDD", "planned_ship_date", "revised_ship_date", "original_ETA", "revised_ETA", "actual_delivery_date",
    "status", "delay_reason",
]
KEYS = ["order_no", "line_no"]


def read_orders_csv(source) -> pd.DataFrame:
    return pd.read_csv(source, parse_dates=ORDER_DATE_COLUMNS, encoding="utf-8-sig", dtype={"order_no": str})


def validate_orders(new: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in new.columns]
    if missing:
        errors.append(f"필요한 컬럼이 없습니다: {', '.join(missing)}")
    elif new.duplicated(KEYS).any():
        dup = new.loc[new.duplicated(KEYS), KEYS].head(3).astype(str).agg("-".join, axis=1).tolist()
        errors.append(f"같은 오더 번호 + 라인이 두 번 이상 있습니다: {', '.join(dup)}")
    elif new["revised_ETA"].isna().any():
        errors.append("revised_ETA(현재 납기)가 비어 있는 행이 있습니다.")
    return errors


def apply_update(new: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """새 오더 파일을 반영: ① 어제 파일과 비교해 ETA 가 바뀐 라인을 이력에 추가 ② 어제 파일 보관 ③ 새 파일·기준일 저장"""
    old = read_orders_csv(DATA_DIR / "orders.csv")
    history = pd.read_csv(DATA_DIR / "eta_history.csv", parse_dates=["snapshot_date", "old_ETA", "new_ETA"], dtype={"order_no": str})

    merged = new.merge(
        old[KEYS + ["revised_ETA", "shipping_mode"]].rename(columns={"revised_ETA": "old_ETA", "shipping_mode": "old_mode"}),
        on=KEYS, how="left",
    )
    changed = merged[merged["old_ETA"].notna() & (merged["revised_ETA"] != merged["old_ETA"])].copy()

    last_seq = history.groupby(KEYS)["change_seq"].max().rename("last_seq")
    changed = changed.merge(last_seq, on=KEYS, how="left")
    air_switch = changed["old_mode"].eq("SEA") & changed["shipping_mode"].eq("AIR")
    pulled_in = changed["revised_ETA"] < changed["old_ETA"]
    reason = changed["delay_reason"].fillna("").replace("", "사유 미기재")
    reason = reason.where(~pulled_in, "공장 일정 앞당김").where(~air_switch, "AIR 전환 (긴급 운송)")
    new_rows = pd.DataFrame({
        "snapshot_date": as_of,
        "order_no": changed["order_no"],
        "line_no": changed["line_no"],
        "change_seq": changed["last_seq"].fillna(0).astype(int) + 1,
        "old_ETA": changed["old_ETA"],
        "new_ETA": changed["revised_ETA"],
        "shift_days": (changed["revised_ETA"] - changed["old_ETA"]).dt.days,
        "reason": reason,
        "shipping_mode": changed["shipping_mode"],
    })

    archive = DATA_DIR / "archive"
    archive.mkdir(exist_ok=True)
    old.to_csv(archive / f"orders_{TODAY:%Y%m%d}.csv", index=False, encoding="utf-8-sig")
    new[[c for c in new.columns if c not in ("delay_days_vs_CRD", "is_delayed")]].to_csv(
        DATA_DIR / "orders.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([history, new_rows], ignore_index=True).to_csv(DATA_DIR / "eta_history.csv", index=False, encoding="utf-8-sig")
    AS_OF_FILE.write_text(f"{as_of:%Y-%m-%d}")

    old_keys = set(map(tuple, old[KEYS].values))
    new_keys = set(map(tuple, new[KEYS].values))
    return {
        "as_of": as_of,
        "changed": len(new_rows),
        "later": int((new_rows["shift_days"] > 0).sum()),
        "earlier": int((new_rows["shift_days"] < 0).sum()),
        "added": len(new_keys - old_keys),
        "removed": len(old_keys - new_keys),
    }


def date_from_filename(name: str):
    match = re.search(r"(20\d{6})", name)
    try:
        return pd.Timestamp(match.group(1)) if match else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 차트 공통
# ---------------------------------------------------------------------------
def style(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=90, b=10),
        title=dict(y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def add_today_line(fig):
    fig.add_shape(type="line", x0=TODAY, x1=TODAY, y0=0, y1=1, yref="paper", line=dict(color=GRAY, width=2, dash="dot"))
    fig.add_annotation(x=TODAY, y=1, yref="paper", text="기준일", showarrow=False, yshift=10, font=dict(color=GRAY))


def hbar(frame, x, y, title, text=None, color=None, color_map=None, order=None, height=360):
    fig = px.bar(
        frame, x=x, y=y, orientation="h", text=text, title=title, color=color,
        color_discrete_sequence=[BLUE], color_discrete_map=color_map,
        category_orders={color: order} if color and order else None, labels={x: "", y: "", color or "": ""},
    )
    if color:
        fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
    else:
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(visible=False, range=[0, frame[x].max() * 1.25 if len(frame) else 1])
    return style(fig, height)


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
orders, history, milestones = load_data(TODAY)

with st.sidebar:
    st.header("보기 모드")
    # 보기 모드는 바로 반영, 필터는 Apply 로 한 번에 반영
    mode_name = st.radio("보고 대상", list(VIEW_MODES), label_visibility="collapsed")
    mode = VIEW_MODES[mode_name]
    st.caption(mode["desc"])

    st.header("필터")
    # 폼 안의 필터는 Apply 를 눌러야 한 번에 반영됨 (선택할 때마다 화면이 다시 그려지지 않음)
    with st.form("filters", border=False):
        projects = st.multiselect("프로젝트", sorted(orders["project_name"].unique()))
        families = st.multiselect("제품군", sorted(orders["product_family"].unique()))
        plants = st.multiselect("공장", sorted(orders["source_plant"].unique()))
        st.form_submit_button("Apply", type="primary", icon=":material/filter_alt:", width="stretch")

    # 오더 파일 업데이트는 내부용 화면에서만
    if mode_name == "전체 (내부용)":
        st.header("데이터 업데이트")
        if "update_result" in st.session_state:
            r = st.session_state.pop("update_result")
            st.success(
                f"{r['as_of']:%Y-%m-%d} 기준으로 반영했습니다. ETA 변경 {r['changed']}건 "
                f"(늦춰짐 {r['later']} · 앞당김 {r['earlier']}), 신규 라인 {r['added']}개, 빠진 라인 {r['removed']}개"
            )
        st.caption(f"현재 데이터 기준일: {TODAY:%Y-%m-%d}")
        # 반영 후에는 업로드 칸을 비우기 위해 key 를 바꿈
        uploaded = st.file_uploader("오더 파일 업로드 (CSV)", type="csv", key=f"orders_upload_{st.session_state.get('upload_count', 0)}",
                                    help="orders.csv 와 같은 컬럼의 CSV. 파일 이름에 날짜(예: orders_20261024.csv)를 넣으면 기준일이 자동으로 채워집니다.")
        if uploaded is not None:
            default_date = date_from_filename(uploaded.name) or TODAY + pd.Timedelta(days=1)
            as_of = pd.Timestamp(st.date_input("이 파일의 기준일", default_date.date(), key="upload_as_of"))
            try:
                new_orders = read_orders_csv(uploaded)
                errors = validate_orders(new_orders)
            except Exception as e:  # 형식이 다른 파일
                new_orders, errors = None, [f"CSV 를 읽을 수 없습니다: {e}"]
            if as_of < TODAY:
                errors.append(f"기준일이 현재 데이터({TODAY:%Y-%m-%d})보다 이전입니다.")
            for err in errors:
                st.error(err)
            if st.button("반영하기", type="primary", icon=":material/upload:", width="stretch", disabled=bool(errors)):
                st.session_state["update_result"] = apply_update(new_orders, as_of)
                st.session_state["upload_count"] = st.session_state.get("upload_count", 0) + 1
                st.cache_data.clear()
                st.rerun()

        with st.expander("데이터 백업"):
            st.caption("Streamlit Cloud 는 재시작하면 서버 파일이 지워지므로, 반영 후 이력 파일을 내려받아 보관하세요.")
            for name in ["orders.csv", "eta_history.csv"]:
                st.download_button(f"📥 {name}", (DATA_DIR / name).read_bytes(), file_name=f"{Path(name).stem}_{TODAY:%Y%m%d}.csv",
                                   mime="text/csv", width="stretch", key=f"backup_{name}")

st.title("📦 판넬 빌더 오더 납기 컨트롤 타워")
st.caption(
    f"고객 {orders['customer_code'].iloc[0]} 프로젝트 오더의 납기 확정·지연·조치 현황 · "
    f"기준일 {TODAY:%Y-%m-%d} · 가상 데이터 · 보기 모드: {mode_name}"
)

df = orders.copy()
if projects:
    df = df[df["project_name"].isin(projects)]
if families:
    df = df[df["product_family"].isin(families)]
if plants:
    df = df[df["source_plant"].isin(plants)]

k = calc_kpis(df)
exc = build_exceptions(df)
tab3_name = "③ 납기 변경 안내" if mode_name == "고객 공유용" else "③ Exception & Action"
# 메일 초안 탭은 내부용 화면에서만 보여줌
show_mail = mode_name == "전체 (내부용)"
tab_names = ["① Overview", "② Project Timeline", tab3_name, "④ 주간 변화"] + (["⑤ 메일 초안"] if show_mail else [])
tabs = st.tabs(tab_names)
tab1, tab2, tab3, tab4 = tabs[:4]

# ---------------------------------------------------------------- ① Overview
with tab1:
    r1 = st.columns(3)
    if mode["amount"]:
        r1[0].metric(
            "Open backlog 금액", won_eok(k["backlog"]), f"{k['open_lines']}개 라인",
            delta_color="off", delta_arrow="off", help="미입고 라인의 잔량 × 단가", border=True,
        )
    else:
        r1[0].metric("미입고 라인", f"{k['open_lines']}개", help="발주 완료·생산 중·운송 중 라인", border=True)
    if mode["internal"]:
        r1[1].metric(
            "OTD% (CRD 대비 / CDD 대비)", f"{k['otd_crd']} / {k['otd_cdd']}", f"입고 완료 {k['delivered_lines']}개 라인",
            delta_color="off", delta_arrow="off",
            help="입고 완료 라인 중 CRD(고객 요청일) / CDD(최초 확정일) 이내에 입고된 비율", border=True,
        )
    else:
        r1[1].metric(
            "요청 납기 준수율", k["otd_crd"], f"입고 완료 {k['delivered_lines']}개 라인",
            delta_color="off", delta_arrow="off", help="입고 완료 라인 중 요청 납기(CRD) 이내에 입고된 비율", border=True,
        )
    r1[2].metric(
        "평균 지연일", f"{k['avg_delay']:.1f}일", f"지연 {k['delayed_lines']}개 라인",
        delta_color="off", delta_arrow="off", help="CRD 보다 늦은 라인의 평균 지연일", border=True,
    )
    r2 = st.columns(3)
    r2[0].metric(
        "분할 납품 비율", k["partial_ratio"],
        help="미입고 라인 중 일부 수량만 출하된 라인 비율", border=True,
    )
    r2[1].metric(
        f"{RISK_WINDOW_DAYS}일 내 위험 오더", f"{k['risk_orders']}건", f"{k['risk_lines']}개 라인",
        delta_color="off", delta_arrow="off", help="CRD 가 30일 안에 도래하지만 현재 ETA 가 CRD 를 넘는 미입고 오더", border=True,
    )

    left, right = st.columns(2)
    with left:
        open_df = df[df["is_open"]]
        if mode["amount"]:
            backlog = (
                open_df.groupby("project_name", as_index=False)["open_amount"].sum()
                .query("open_amount > 0").sort_values("open_amount")
            )
            backlog["label"] = backlog["open_amount"].map(won_eok)
            fig = hbar(backlog, "open_amount", "project_name", "프로젝트별 Open backlog", text="label")
            fig.update_traces(hovertemplate="%{y}<br>%{text}<extra></extra>")
        elif mode["supply"]:
            by_plant = (
                df[df["is_open"] & df["is_delayed"]].groupby(["source_plant", "shipping_mode"], as_index=False).size()
            )
            fig = hbar(by_plant, "size", "source_plant", "공장·운송 방식별 지연 라인", color="shipping_mode",
                       color_map={"SEA": BLUE, "AIR": ORANGE}, order=["SEA", "AIR"])
        else:
            by_status = open_df.groupby(["project_name", "status"], as_index=False).size()
            fig = hbar(by_status, "size", "project_name", "프로젝트별 미입고 라인 (진행 단계별)", color="status",
                       color_map={"발주 완료": GRAY, "생산 중": BLUE, "운송 중": ORANGE}, order=OPEN_STATUSES)
        st.plotly_chart(fig, width="stretch")

    with right:
        delivered = df[df["status"] == "입고 완료"].copy()
        if delivered.empty:
            # 아직 입고 완료된 라인이 없는 프로젝트(예: 세종 물류센터)는 추이를 그릴 수 없음
            st.markdown("**월별 OTD% 추이**" if mode["internal"] else "**월별 요청 납기 준수율**")
            st.info("선택한 조건에 입고 완료된 라인이 아직 없습니다.")
        else:
            delivered["month"] = delivered["actual_delivery_date"].dt.to_period("M").dt.to_timestamp()
            delivered["CRD 대비"] = delivered["actual_delivery_date"] <= delivered["CRD"]
            delivered["CDD 대비"] = delivered["actual_delivery_date"] <= delivered["CDD"]
            trend = delivered.groupby("month", as_index=False).agg(
                **{"CRD 대비": ("CRD 대비", "mean"), "CDD 대비": ("CDD 대비", "mean"), "입고 라인 수": ("line_key", "count")}
            )
            series = ["CRD 대비", "CDD 대비"] if mode["internal"] else ["CRD 대비"]
            trend_long = trend.melt(id_vars=["month", "입고 라인 수"], value_vars=series, var_name="기준", value_name="OTD")
            fig = px.line(
                trend_long, x="month", y="OTD", color="기준", markers=True,
                title="월별 OTD% 추이" if mode["internal"] else "월별 요청 납기 준수율",
                color_discrete_map={"CRD 대비": BLUE, "CDD 대비": ORANGE}, hover_data={"입고 라인 수": True},
                labels={"month": "입고 월", "OTD": "OTD%"},
            )
            fig.update_traces(line_width=2, marker_size=8)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
            fig.update_xaxes(dtick="M1", tickformat="%Y-%m")
            if not mode["internal"]:
                fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig), width="stretch")

    delayed_df = df[df["is_delayed"] & (df["delay_reason"].fillna("") != "")]
    if mode["internal"]:
        if mode["supply"]:
            reasons = delayed_df.groupby(["delay_reason", "source_plant"], as_index=False).size()
            fig = hbar(reasons, "size", "delay_reason", "지연 사유별 라인 수 (공장별)", color="source_plant",
                       color_map={"Singapore": BLUE, "France": ORANGE}, order=["Singapore", "France"], height=300)
        else:
            reasons = delayed_df.groupby("delay_reason", as_index=False).size().sort_values("size")
            fig = hbar(reasons, "size", "delay_reason", "지연 사유별 라인 수", text="size", height=300)
    else:
        reasons = (
            delayed_df.assign(customer_reason=delayed_df["delay_reason"].map(CUSTOMER_REASON))
            .groupby("customer_reason", as_index=False).size().sort_values("size")
        )
        fig = hbar(reasons, "size", "customer_reason", "납기 변경 사유별 라인 수", text="size", height=300)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- ② Project Timeline
with tab2:
    ms_projects = sorted(milestones["project_name"].unique())
    if projects:
        ms_projects = [p for p in ms_projects if p in projects]
    # 처음 열었을 때는 설치 완료일이 가장 많이 밀린 서브 프로젝트를 보여줌
    site_end = milestones[milestones["phase"].str.startswith("5.")].pivot_table(
        index=["project_name", "sub_project_name"], columns="plan_type", values="end"
    )
    site_shift = (site_end["현재 예상"] - site_end["고객 계획"]).dt.days
    site_shift = site_shift[site_shift.index.get_level_values(0).isin(ms_projects)]
    worst_project, worst_sub = site_shift.idxmax()

    s1, s2 = st.columns(2)
    project = s1.selectbox("프로젝트", ms_projects, index=ms_projects.index(worst_project))
    sub_options = sorted(milestones.loc[milestones["project_name"] == project, "sub_project_name"].unique())
    sub = s2.selectbox("서브 프로젝트", sub_options, index=sub_options.index(worst_sub) if project == worst_project else 0)

    ms = milestones[(milestones["project_name"] == project) & (milestones["sub_project_name"] == sub)].copy()
    plan_end = ms.loc[ms["plan_type"] == "고객 계획", "end"].max()
    forecast_end = ms.loc[ms["plan_type"] == "현재 예상", "end"].max()
    shift = (forecast_end - plan_end).days
    gating = ms.loc[ms["gating_line"].fillna("") != "", "gating_line"].iloc[0]

    t1, t2, t3 = st.columns(3)
    t1.metric("고객 계획 설치 완료", f"{plan_end:%Y-%m-%d}")
    t2.metric("현재 예상 설치 완료", f"{forecast_end:%Y-%m-%d}", f"{shift:+d}일", delta_color="inverse")
    t3.metric("자재 공급 병목 라인", gating.split(" ", 1)[0], gating.split(" ", 1)[1], delta_color="off", delta_arrow="off")

    ms["기간(일)"] = (ms["end"] - ms["start"]).dt.days
    fig = px.timeline(
        ms, x_start="start", x_end="end", y="phase", color="plan_type",
        color_discrete_map={"고객 계획": GRAY, "현재 예상": BLUE},
        category_orders={"plan_type": ["고객 계획", "현재 예상"]},
        hover_data={"start": "|%Y-%m-%d", "end": "|%Y-%m-%d", "기간(일)": True, "phase": False, "plan_type": False},
        title=f"{project} · {sub} 단계별 일정 (고객 계획 vs 현재 예상)",
        labels={"phase": "", "plan_type": "", "start": "시작", "end": "종료"},
    )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", tickformat="%Y-%m")
    add_today_line(fig)
    st.plotly_chart(style(fig, height=420), width="stretch")

    if mode["line_detail"]:
        st.markdown("**자재 공급 라인 (입고 예정일 늦은 순)**")
        sub_lines = df[(df["project_name"] == project) & (df["sub_project_name"] == sub) & (df["status"] != "취소")].copy()
        sub_lines["material_date"] = sub_lines["actual_delivery_date"].fillna(sub_lines["revised_ETA"])
        line_cols = {
            "line_key": "오더-라인", "product_family": "제품군", "source_plant": "공장", "shipping_mode": "운송",
            "status": "상태", "CRD": "CRD", "material_date": "입고(예정)일", "delay_days_vs_CRD": "CRD 대비 지연일",
            "eta_change_count": "ETA 변경 횟수",
        }
        if not mode["internal"]:
            line_cols.pop("source_plant")
        st.dataframe(
            table(sub_lines.sort_values("material_date", ascending=False).head(10), line_cols),
            hide_index=True, width="stretch",
        )

    # AIR 전환 시뮬레이션 (내부 화면에서만)
    if mode["internal"]:
        st.divider()
        st.subheader("✈️ AIR 전환 시뮬레이션")
        st.caption("출하 전인 SEA 라인을 AIR(출하 + 10일)로 바꾸면 자재 입고와 이후 단계가 얼마나 당겨지는지 계산합니다. 필터와 관계없이 서브 프로젝트의 모든 라인으로 계산합니다.")

        sim_lines = orders[(orders["project_name"] == project) & (orders["sub_project_name"] == sub) & (orders["status"] != "취소")].copy()
        sim_lines["material_date"] = sim_lines["actual_delivery_date"].fillna(sim_lines["revised_ETA"])
        cand = air_candidates(sim_lines)

        if cand.empty:
            st.info("이 서브 프로젝트에는 AIR 로 바꿀 수 있는 라인(출하 전 SEA)이 없습니다.")
        else:
            goal = st.radio(
                "목표", ["고객 계획 일정 맞추기", "최대한 당기기"], horizontal=True, key=f"air_goal_{project}_{sub}",
                help="목표를 이루는 데 필요한 최소 라인 조합을 자동으로 선택합니다. 아래에서 직접 바꿀 수도 있습니다.",
            )
            # 고객 계획은 모든 자재가 CRD 에 입고된다고 가정한 일정이므로, 자재 입고 완료 목표 = 가장 늦은 CRD
            target = sim_lines["CRD"].max() if goal == "고객 계획 일정 맞추기" else None
            recommended = minimal_air_set(sim_lines, cand, target)
            cand = cand.sort_values("revised_ETA", ascending=False)
            picked = st.multiselect(
                "AIR 로 바꿀 라인", cand.index.tolist(), default=recommended,
                format_func=lambda i: (
                    f"{cand.at[i, 'line_key']} {cand.at[i, 'product_family']} · "
                    f"{cand.at[i, 'revised_ETA']:%m-%d} → {cand.at[i, 'air_eta']:%m-%d} "
                    f"(-{(cand.at[i, 'revised_ETA'] - cand.at[i, 'air_eta']).days}일)"
                ),
                key=f"air_sim_{project}_{sub}_{goal}",
            )

            simulated = sim_lines["material_date"].copy()
            simulated[picked] = cand.loc[picked, "air_eta"]
            fab_days = 45 if "BlokSeT" in sub else 30
            order_start, order_end = sim_lines["order_date"].min(), sim_lines["order_date"].max()
            sim_ms = pd.concat([
                ms[["phase", "start", "end", "plan_type"]],
                phase_schedule(order_start, order_end, simulated.max(), fab_days, "AIR 전환 시"),
            ])
            sim_end = sim_ms.loc[sim_ms["plan_type"] == "AIR 전환 시", "end"].max()
            new_gating = sim_lines.loc[simulated.idxmax()]

            g1, g2, g3, g4 = st.columns([0.9, 1.2, 0.9, 1.5])
            g1.metric("AIR 전환 라인", f"{len(picked)}개", f"후보 {len(cand)}개 중", delta_color="off", delta_arrow="off")
            g2.metric("AIR 전환 시 설치 완료", f"{sim_end:%Y-%m-%d}", f"{(sim_end - forecast_end).days:+d}일 (현재 예상 대비)", delta_color="inverse")
            g3.metric("고객 계획 대비", f"{(sim_end - plan_end).days:+d}일", "계획 이내" if sim_end <= plan_end else "계획 초과",
                      delta_color="off", delta_arrow="off")
            g4.metric("전환 후 병목 라인", new_gating["line_key"], new_gating["product_family"], delta_color="off", delta_arrow="off")

            sim_ms["기간(일)"] = (sim_ms["end"] - sim_ms["start"]).dt.days
            plan_order = ["고객 계획", "현재 예상", "AIR 전환 시"]
            fig = px.timeline(
                sim_ms, x_start="start", x_end="end", y="phase", color="plan_type",
                color_discrete_map={"고객 계획": GRAY, "현재 예상": BLUE, "AIR 전환 시": ORANGE},
                category_orders={"plan_type": plan_order},
                hover_data={"start": "|%Y-%m-%d", "end": "|%Y-%m-%d", "기간(일)": True, "phase": False, "plan_type": False},
                title="고객 계획 vs 현재 예상 vs AIR 전환 시",
                labels={"phase": "", "plan_type": "", "start": "시작", "end": "종료"},
            )
            fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", tickformat="%Y-%m")
            add_today_line(fig)
            st.plotly_chart(style(fig, height=480), width="stretch")
            st.caption("AIR 운임 등 추가 비용은 포함되지 않았습니다. 전환 전 공급망팀과 비용·가용 여부를 확인하세요.")

            # AIR 로 바꿀 라인 엑셀 다운로드
            air_lines = cand.loc[picked].copy()
            air_lines["saving_days"] = (air_lines["revised_ETA"] - air_lines["air_eta"]).dt.days
            air_table = table(air_lines.sort_values("saving_days", ascending=False), {
                "order_no": "오더 번호", "line_no": "라인", "project_name": "프로젝트", "sub_project_name": "서브 프로젝트",
                "product_family": "제품군", "product_code": "제품 코드", "source_plant": "공장", "quantity": "수량",
                "status": "상태", "revised_ship_date": "출하 예정일", "CRD": "CRD", "revised_ETA": "현재 ETA (SEA)",
                "air_eta": "AIR 전환 시 ETA", "saving_days": "단축일", "delay_days_vs_CRD": "현재 CRD 대비 지연일",
            })
            sim_summary = pd.DataFrame({
                "항목": ["프로젝트", "서브 프로젝트", "목표", "AIR 전환 라인 수", "고객 계획 설치 완료",
                       "현재 예상 설치 완료", "AIR 전환 시 설치 완료", "현재 예상 대비 단축일", "전환 후 병목 라인"],
                "값": [project, sub, goal, len(picked), f"{plan_end:%Y-%m-%d}", f"{forecast_end:%Y-%m-%d}",
                      f"{sim_end:%Y-%m-%d}", (forecast_end - sim_end).days,
                      f"{new_gating['line_key']} {new_gating['product_family']}"],
            })
            st.download_button(
                "📥 AIR 전환 라인 엑셀 다운로드",
                data=to_excel({"AIR 전환 라인": air_table, "시뮬레이션 요약": sim_summary}),
                file_name=f"AIR전환_{project}_{sub}_{TODAY:%Y%m%d}.xlsx".replace(" ", ""),
                mime=XLSX_MIME, disabled=not picked, key="air_download",
            )

# ---------------------------------------------------------------- ③ Exception & Action / 납기 변경 안내
with tab3:
    buckets = ["CRD 경과", f"{RISK_WINDOW_DAYS}일 내 도래", f"{RISK_WINDOW_DAYS}일 이후"]

    if mode_name == "고객 공유용":
        st.caption("요청 납기(CRD)보다 늦어질 것으로 예상되는 미입고 오더와 변경 납기 예정일입니다.")
        crd_sel = st.multiselect("요청 납기 도래", buckets, default=buckets)
        view = exc[exc["crd_bucket"].isin(crd_sel)]
        c1, c2 = st.columns(2)
        c1.metric("납기 변경 라인", f"{len(view)}개", f"{view['order_no'].nunique()}개 오더", delta_color="off", delta_arrow="off")
        c2.metric("평균 변경 일수", f"{view['delay_days_vs_CRD'].mean():.1f}일" if len(view) else "-")
        st.dataframe(table(view, CUSTOMER_COLUMNS), hide_index=True, width="stretch", height=420)
    else:
        st.caption("미입고 라인 중 현재 ETA 가 CRD 를 넘는 라인입니다. 심각도는 CRD 대비 지연일 기준(30일 이상 심각, 8-29일 주의, 1-7일 경미)입니다.")
        f1, f2, f3 = st.columns([1.2, 1.2, 1])
        sev = f1.multiselect("심각도", list(SEVERITY_COLORS), default=list(SEVERITY_COLORS))
        crd_sel = f2.multiselect("CRD 도래", buckets, default=buckets)
        # 공급망팀용은 AIR 전환 후보를 먼저 보여줌
        only_air = f3.toggle("AIR 전환 가능 건만 보기", value=mode["supply"])
        view = exc[exc["severity"].isin(sev) & exc["crd_bucket"].isin(crd_sel)]
        if only_air:
            view = view[view["air_saving_days"] > 0]

        a1, a2, a3 = st.columns(3)
        a1.metric("지연·위험 라인", f"{len(view)}개", f"{view['order_no'].nunique()}개 오더", delta_color="off", delta_arrow="off")
        if mode["amount"]:
            a2.metric("관련 backlog", won_eok(view["open_amount"].sum()))
        else:
            a2.metric("평균 지연일", f"{view['delay_days_vs_CRD'].mean():.1f}일" if len(view) else "-")
        air = view[view["air_saving_days"] > 0]
        a3.metric("AIR 전환 가능", f"{len(air)}개 라인", f"평균 {air['air_saving_days'].mean():.0f}일 단축" if len(air) else None, delta_color="off", delta_arrow="off")

        group_col = "source_plant" if mode["supply"] else "project_name"
        by_group = view.groupby([group_col, "severity"], as_index=False).size()
        fig = hbar(by_group, "size", group_col, f"{'공장' if mode['supply'] else '프로젝트'}별 지연·위험 라인 (심각도별)",
                   color="severity", color_map=SEVERITY_COLORS, order=list(SEVERITY_COLORS), height=320)
        st.plotly_chart(fig, width="stretch")

        if mode["line_detail"]:
            columns = SUPPLY_COLUMNS if mode["supply"] else FULL_COLUMNS
            st.dataframe(
                table(view, columns), hide_index=True, width="stretch", height=420,
                column_config={"조치 제안": st.column_config.TextColumn(width="large")},
            )
        else:
            st.markdown("**프로젝트 요약**")
            st.dataframe(project_summary(view), hide_index=True, width="stretch")
            st.markdown("**심각 지연 TOP 10**")
            st.dataframe(table(view[view["severity"] == "심각"].head(10), MANAGER_TOP_COLUMNS),
                         hide_index=True, width="stretch",
                         column_config={"잔량 금액(원)": st.column_config.NumberColumn(format="localized")})

    sheets, file_prefix = build_report(mode_name, view, history)
    st.download_button(
        f"📥 엑셀 다운로드 ({mode_name})",
        data=to_excel(sheets),
        file_name=f"{file_prefix}_{TODAY:%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=view.empty,
    )
    st.caption(f"엑셀 시트 구성: {', '.join(sheets)}")

# ---------------------------------------------------------------- ④ 주간 변화
with tab4:
    w = weekly_changes(df, history)
    new_delay, resolved, changes = w["new_delay"], w["resolved"], w["changes"].copy()
    customer = mode_name == "고객 공유용"
    changes["customer_reason"] = changes["reason"].map(CUSTOMER_REASON).fillna("")

    st.caption(
        f"이번 주({WEEK_START:%m-%d} ~ {TODAY:%m-%d})와 지난주 기준일({LAST_WEEK:%m-%d}) 시점을 비교합니다. "
        "지난주 시점의 ETA 는 ETA 변경 이력으로 복원했습니다."
    )
    m = st.columns(3) + st.columns(3)[:2]
    m[0].metric(
        "지연 라인", f"{w['delayed_now']}개", f"{w['delayed_now'] - w['delayed_then']:+d}개 (지난주 {w['delayed_then']}개)",
        delta_color="inverse", border=True,
    )
    by_cause = new_delay["cause"].value_counts()
    m[1].metric(
        "신규 지연", f"{len(new_delay)}개",
        f"ETA 변경 {by_cause.get('ETA 변경', 0)} · 신규 발주 {by_cause.get('이번 주 신규 발주', 0)}",
        delta_color="off", delta_arrow="off", border=True,
    )
    m[2].metric("지연 해소", f"{len(resolved)}개", help="지난주에는 CRD 를 넘었지만 지금은 CRD 안으로 들어온 라인", border=True)
    later, earlier = int((changes["shift_days"] > 0).sum()), int((changes["shift_days"] < 0).sum())
    m[3].metric(
        "ETA 변경", f"{changes['line_key'].nunique()}개 라인", f"늦춰짐 {later} · 앞당김 {earlier}",
        delta_color="off", delta_arrow="off", border=True,
    )
    m[4].metric(
        "이번 주 입고 완료", f"{len(w['delivered'])}개 라인", f"CRD 이후 입고 {w['delivered_late']}개",
        delta_color="off", delta_arrow="off", border=True,
    )

    trend = weekly_change_trend(df, history)
    fig = px.bar(
        trend, x="week_end", y="size", color="direction", title="주별 ETA 변경 건수 (최근 8주)",
        color_discrete_map={"늦춰짐": ORANGE, "앞당김": BLUE}, category_orders={"direction": ["늦춰짐", "앞당김"]},
        labels={"week_end": "주 마감일", "size": "변경 건수", "direction": ""},
    )
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
    fig.update_xaxes(type="category")
    st.plotly_chart(style(fig, height=320), width="stretch")

    # 모드별 표 구성
    if customer:
        new_cols = {
            "order_no": "오더 번호", "line_no": "라인", "project_name": "프로젝트", "sub_project_name": "서브 프로젝트",
            "product_family": "제품군", "product_code": "제품 코드", "CRD": "요청 납기(CRD)", "eta_last_week": "기존 안내 납기",
            "revised_ETA": "변경 납기 예정일", "shift_this_week": "이번 주 변동일", "customer_reason": "변경 사유",
        }
        change_cols = {
            "snapshot_date": "변경일", "line_key": "오더-라인", "project_name": "프로젝트", "product_family": "제품군",
            "old_ETA": "기존 납기", "new_ETA": "변경 납기", "shift_days": "변동 일수", "customer_reason": "변경 사유",
        }
    else:
        new_cols = {
            "line_key": "오더-라인", "project_name": "프로젝트", "sub_project_name": "서브 프로젝트", "product_family": "제품군",
            "source_plant": "공장", "shipping_mode": "운송", "cause": "구분", "CRD": "CRD", "eta_last_week": "지난주 ETA",
            "revised_ETA": "현재 ETA", "shift_this_week": "이번 주 변동일", "delay_days_vs_CRD": "CRD 대비 지연일",
            "delay_reason": "지연 사유",
        }
        change_cols = {
            "snapshot_date": "변경 통보일", "line_key": "오더-라인", "project_name": "프로젝트", "product_family": "제품군",
            "old_ETA": "이전 ETA", "new_ETA": "변경 ETA", "shift_days": "변동 일수", "reason": "사유", "shipping_mode": "운송",
        }
    resolved_cols = {
        "line_key": "오더-라인", "project_name": "프로젝트", "product_family": "제품군", "CRD": "CRD",
        "eta_last_week": "지난주 ETA", "revised_ETA": "현재 ETA", "shipping_mode": "운송",
    }

    st.markdown(f"**{'이번 주 납기가 변경된 오더' if customer else '이번 주 신규 지연 라인'}** ({len(new_delay)}개)")
    new_table = table(new_delay if mode["line_detail"] else new_delay.head(10), new_cols)
    st.dataframe(new_table, hide_index=True, width="stretch", height=min(38 * (len(new_table) + 1), 420))

    st.markdown(f"**{'납기가 정상화된 오더' if customer else '지연 해소 라인'}** ({len(resolved)}개)")
    if resolved.empty:
        st.caption("이번 주에 해소된 지연이 없습니다.")
    else:
        st.dataframe(table(resolved, resolved_cols), hide_index=True, width="stretch")

    if mode["line_detail"]:
        st.markdown(f"**이번 주 ETA 변경 내역** ({len(changes)}건)")
        st.dataframe(table(changes, change_cols), hide_index=True, width="stretch", height=360)

    weekly_sheets = {
        "신규 지연" if not customer else "납기 변경 오더": table(new_delay, new_cols),
        "지연 해소" if not customer else "납기 정상화 오더": table(resolved, resolved_cols),
        "ETA 변경 내역": table(changes, change_cols),
    }
    st.download_button(
        f"📥 주간 변화 엑셀 다운로드 ({mode_name})",
        data=to_excel(weekly_sheets),
        file_name=f"PB_weekly_changes_{TODAY:%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="weekly_download",
    )

# ---------------------------------------------------------------- ⑤ 메일 초안
if show_mail:
    with tabs[4]:
        sender = st.text_input("보내는 사람 이름", placeholder="예: 홍길동", key="mail_sender") or "OOO"
        mail_types = ["고객 납기 회신", "공급망 담당자 단축 요청"]
        mail_type = st.segmented_control("메일 종류", mail_types, default=mail_types[0], key="mail_type") or mail_types[0]

        if mail_type == "고객 납기 회신":
            open_lines = df[df["is_open"]]
            mail_projects = sorted(open_lines["project_name"].unique())
            if not mail_projects:
                st.info("미입고 오더가 없습니다.")
            else:
                m1, m2 = st.columns(2)
                mail_project = m1.selectbox("프로젝트", mail_projects, key="mail_project")
                to = m2.text_input("받는 사람 이메일", placeholder="customer@example.com", key="mail_customer_to")

                lines = open_lines[open_lines["project_name"] == mail_project].sort_values(["sub_project_name", "order_no", "line_no"])
                attach_table = table(lines, CUSTOMER_SCHEDULE_COLUMNS)
                filename = f"{mail_project}_납기일정_{TODAY:%Y%m%d}.xlsx"
                subject, body = customer_mail(mail_project, sender)

                st.markdown(f"**제목** {subject}")
                st.code(body, language=None, wrap_lines=True)
                st.markdown(f"**첨부** {filename} · 미입고 {len(lines)}개 라인 ({lines['order_no'].nunique()}개 오더)")
                st.dataframe(attach_table, hide_index=True, width="stretch", height=min(35 * (len(attach_table) + 1) + 3, 300))
                mail_actions(to, subject, body, to_excel({"납기 일정": attach_table}), filename, "customer_mail")

        else:
            if exc.empty:
                st.info("지연 오더가 없습니다.")
            else:
                m1, m2, m3 = st.columns(3)
                delay_projects = ["전체"] + sorted(exc["project_name"].unique())
                delay_project = m1.selectbox("프로젝트", delay_projects, key="supply_project")
                manager = m2.text_input("공급망 담당자 이름", placeholder="예: 김OO", key="supply_manager") or "OOO"
                to = m3.text_input("받는 사람 이메일", placeholder="supply@example.com", key="supply_to")

                candidates = exc if delay_project == "전체" else exc[exc["project_name"] == delay_project]
                # 지연이 큰 오더부터
                order_options = candidates.groupby("order_no")["delay_days_vs_CRD"].max().sort_values(ascending=False).index.tolist()
                # 전체 프로젝트일 때는 30일 이상 지연(심각) 오더만 기본 선택
                max_delay = candidates.groupby("order_no")["delay_days_vs_CRD"].max()
                default_orders = order_options if delay_project != "전체" else [o for o in order_options if max_delay[o] >= 30]
                picked = st.multiselect(
                    "요청할 지연 오더", order_options, default=default_orders,
                    format_func=lambda o: f"{o} (최대 {candidates.loc[candidates['order_no'] == o, 'delay_days_vs_CRD'].max()}일 지연)",
                    key=f"supply_orders_{delay_project}",
                )
                if not picked:
                    st.info("요청할 오더를 하나 이상 선택해 주세요.")
                else:
                    lines = candidates[candidates["order_no"].isin(picked)].copy()
                    lines["request"] = lines.apply(supply_request, axis=1)
                    lines = lines.sort_values(["delay_days_vs_CRD", "order_no", "line_no"], ascending=[False, True, True])
                    attach_table = table(lines, SUPPLY_REQUEST_COLUMNS)
                    filename = f"납기확인_단축요청_{TODAY:%Y%m%d}.xlsx"
                    subject, body = supply_mail(picked, manager, sender)

                    st.markdown(f"**제목** {subject}")
                    st.code(body, language=None, wrap_lines=True)
                    st.markdown(f"**첨부** {filename} · 지연 {len(lines)}개 라인 ({len(picked)}개 오더)")
                    st.dataframe(attach_table, hide_index=True, width="stretch", height=min(35 * (len(attach_table) + 1) + 3, 300))
                    mail_actions(to, subject, body, to_excel({"납기 확인 및 단축 요청": attach_table}), filename, "supply_mail")
