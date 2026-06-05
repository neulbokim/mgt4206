# ./data/event_scraper.py

import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE_URL = "https://www.kh.or.kr"
LIST_URL = f"{BASE_URL}/program/list/menu/1292"
VIEW_URL = f"{BASE_URL}/program/view/menu/1292"

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": LIST_URL,
}


def clean_text(text: str) -> str:
    """공백, 줄바꿈 정리"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_soup(url: str, params: dict | None = None) -> BeautifulSoup:
    res = requests.get(url, params=params, headers=HEADERS, timeout=15)
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


def extract_rounds(soup: BeautifulSoup) -> list[dict]:
    """
    역대 프로그램 회차 추출
    예: searchValues('ach','26')
    """
    rounds = []

    for a in soup.select("a[onclick*='searchValues']"):
        onclick = a.get("onclick", "")
        m = re.search(r"searchValues\('ach','(\d+)'\)", onclick)
        if not m:
            continue

        round_id = m.group(1)
        round_label = clean_text(a.get_text())

        item = {
            "round_id": round_id,
            "round_label": round_label,
        }

        if item not in rounds:
            rounds.append(item)

    return rounds


def extract_program_idx(onclick: str) -> str:
    """
    예: goView('637') -> 637
    """
    if not onclick:
        return ""

    m = re.search(r"goView\('(\d+)'\)", onclick)
    return m.group(1) if m else ""


def parse_list_page(round_id: str, search_text: str = "") -> list[dict]:
    """
    특정 회차의 목록 페이지에서 프로그램 카드 정보 추출
    """
    params = {
        "viewType": "ach",
        "idx": "",
        "searchRound": round_id,
        "searchStTime": "",
        "searchEnTime": "",
        "searchField": "homeAll",
        "searchText": search_text,
    }

    soup = get_soup(LIST_URL, params=params)

    programs = []

    for area in soup.select(".thumnail_area"):
        palace = clean_text(area.select_one(".thumnail_tit").get_text()) if area.select_one(".thumnail_tit") else ""

        for card in area.select(".thumnail_list"):
            title_tag = card.select_one(".cont_top a")
            if not title_tag:
                continue

            onclick = title_tag.get("onclick", "")
            program_idx = extract_program_idx(onclick)

            title = clean_text(title_tag.get_text())

            type_tag = card.select_one(".cont_top span")
            program_type = clean_text(type_tag.get_text()) if type_tag else ""

            date_tag = card.select_one(".cont_date p")
            schedule_summary = clean_text(date_tag.get_text()) if date_tag else ""

            img_tag = card.select_one("img")
            img_url = ""
            img_alt = ""

            if img_tag:
                img_src = img_tag.get("src", "")
                img_url = urljoin(BASE_URL, img_src)
                img_alt = clean_text(img_tag.get("alt", ""))

            programs.append({
                "round_id": round_id,
                "palace_group": palace,
                "program_idx": program_idx,
                "title": title,
                "program_type_from_list": program_type,
                "schedule_summary": schedule_summary,
                "image_url_from_list": img_url,
                "image_alt": img_alt,
            })

    return programs


def parse_detail_page(program_idx: str, round_id: str, search_text: str = "") -> dict:
    """
    상세 페이지에서 상태, 유형, 참여유형, 기간, 시간, 장소, 상세내용 추출
    """
    params = {
        "viewType": "ach",
        "idx": program_idx,
        "searchRound": round_id,
        "searchStTime": "",
        "searchEnTime": "",
        "searchField": "homeAll",
        "searchText": search_text,
    }

    soup = get_soup(VIEW_URL, params=params)

    result = {
        "detail_url": requests.Request("GET", VIEW_URL, params=params).prepare().url,
        "detail_palace": "",
        "detail_title": "",
        "detail_category_in_title": "",
        "status": "",
        "program_type": "",
        "participation_type": "",
        "period": "",
        "time": "",
        "place": "",
        "capacity": "",
        "contact": "",
        "detail_text": "",
        "detail_image_url": "",
    }

    top_tag = soup.select_one(".program_tit .top")
    if top_tag:
        result["detail_palace"] = clean_text(top_tag.get_text())

    title_tag = soup.select_one(".program_tit .tit")
    if title_tag:
        raw_title = clean_text(title_tag.get_text())

        category_match = re.search(r"\[(.*?)\]", raw_title)
        if category_match:
            result["detail_category_in_title"] = category_match.group(1)

        result["detail_title"] = clean_text(re.sub(r"\[.*?\]", "", raw_title))

    img_tag = soup.select_one(".program_area .img_area img")
    if img_tag:
        result["detail_image_url"] = urljoin(BASE_URL, img_tag.get("src", ""))

    # 좌측 상세 정보 ul.program_cont li
    for li in soup.select(".program_cont li"):
        span = li.select_one("span")
        if not span:
            continue

        key = clean_text(span.get_text())
        value = clean_text(li.get_text().replace(span.get_text(), "", 1))

        if key == "상태":
            result["status"] = value
        elif key == "유형":
            result["program_type"] = value
        elif key == "참여 유형":
            result["participation_type"] = value
        elif key == "기간":
            result["period"] = value
        elif key == "시간":
            result["time"] = value
        elif key == "장소":
            result["place"] = value
        elif key == "인원":
            result["capacity"] = value
        elif key == "문의":
            result["contact"] = value

    detail_tag = soup.select_one(".view_con")
    if detail_tag:
        result["detail_text"] = clean_text(detail_tag.get_text(separator="\n"))

    return result


def scrape_all(search_text: str = "", sleep_sec: float = 0.3) -> pd.DataFrame:
    """
    전체 역대 회차를 순회하면서 목록 + 상세 데이터 수집
    """
    first_soup = get_soup(
        LIST_URL,
        params={
            "viewType": "ach",
            "idx": "",
            "searchRound": "",
            "searchStTime": "",
            "searchEnTime": "",
            "searchField": "homeAll",
            "searchText": search_text,
        },
    )

    rounds = extract_rounds(first_soup)

    print(f"[INFO] 발견한 회차 수: {len(rounds)}")
    print(rounds)

    all_rows = []

    for round_info in rounds:
        round_id = round_info["round_id"]
        round_label = round_info["round_label"]

        print(f"\n[INFO] 회차 수집 중: {round_label} / round_id={round_id}")

        list_items = parse_list_page(round_id=round_id, search_text=search_text)
        print(f"  - 목록 프로그램 수: {len(list_items)}")

        for i, item in enumerate(list_items, start=1):
            program_idx = item.get("program_idx", "")

            if not program_idx:
                row = {
                    **round_info,
                    **item,
                }
                all_rows.append(row)
                continue

            try:
                detail = parse_detail_page(
                    program_idx=program_idx,
                    round_id=round_id,
                    search_text=search_text,
                )

                row = {
                    **round_info,
                    **item,
                    **detail,
                }

                all_rows.append(row)

                print(f"    [{i}/{len(list_items)}] {item['title']} 완료")

                time.sleep(sleep_sec)

            except Exception as e:
                print(f"    [{i}/{len(list_items)}] {item['title']} 실패: {e}")

                row = {
                    **round_info,
                    **item,
                    "error": str(e),
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # 중복 제거: 같은 회차 + 프로그램 idx 기준
    if not df.empty and "program_idx" in df.columns:
        df = df.drop_duplicates(subset=["round_id", "program_idx"], keep="first")

    return df


def save_outputs(df: pd.DataFrame, prefix: str = "kh_royal_culture_events") -> None:
    csv_path = RAW_DIR / f"{prefix}.csv"
    json_path = RAW_DIR / f"{prefix}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            df.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[INFO] 저장 완료")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"총 {len(df)}개 프로그램 저장")


if __name__ == "__main__":
    # 전체 역대 행사 수집
    df_events = scrape_all(search_text="", sleep_sec=0.3)
    save_outputs(df_events)

    # 경복궁 관련 행사만 수집하고 싶으면 위 2줄 대신 아래 2줄 사용
    # df_events = scrape_all(search_text="경복궁", sleep_sec=0.3)
    # save_outputs(df_events, prefix="kh_royal_culture_events_gyeongbokgung")