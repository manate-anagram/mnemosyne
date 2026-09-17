"""CJK canonical-slot gating regressions for #971.

Spaceless CJK text used to be tokenized one character at a time, so any two
characters shared with a canonical slot satisfied the "two distinctive tokens"
gate. Unrelated single-source-of-truth facts therefore scored 0.94-0.98 and
were injected into the memory context of every turn, crowding out precise
recall.

These tests pin the 2-gram behaviour on the public canonical paths:

* unrelated CJK queries return no canonical rows,
* genuine overlaps are preserved for Japanese, Korean and Chinese,
* a short question whose only topical unit is its last 2-gram still matches
  (the Chinese "什么时候部署？" case that a plain unit-count threshold dropped),
* a one-character CJK run still matches, so very short slots keep working,
* Latin behaviour is untouched.
"""

from __future__ import annotations

import pytest

from mnemosyne_hermes import _canonical_prefetch_rows, _canonical_recall_rows

JAPANESE = {
    "slots": [
        ("identity", "name_pronoun", "利用者は「タナカ」と呼び、一人称は「私」を使う。"),
        ("preference", "tea", "静かな喫茶店と日本茶（緑茶・ほうじ茶）を好む。"),
        ("procedure", "backup", "週次のバックアップ手順を実行し、保存先は外付けディスクとする。"),
        ("environment", "shared_server", "社内の共有サーバーは業務時間内のみ稼働する。"),
        ("fact", "inventory", "棚卸しは月末の金曜日にまとめて数える。"),
        ("procedure", "deploy", "配備は金曜日を避け、火曜日の午前中に行う。"),
    ],
    "unrelated": [
        "会議室の予約ルールはどこにある？",
        "ノートパソコンの在庫は足りてる？",
        "新入社員の研修はいつから？",
        "備品の注文は誰が承認する？",
        "印刷機のトナーはどこに置いてある？",
        "駐車場の契約を更新したい",
    ],
    "related": [
        ("静かな喫茶店を探している", "tea"),
        ("バックアップの保存先は？", "backup"),
        ("共有サーバーの稼働時間は？", "shared_server"),
        ("棚卸しの日程は？", "inventory"),
        ("配備はいつ実施する？", "deploy"),
    ],
}

KOREAN = {
    "slots": [
        ("identity", "name_pronoun", "사용자를 철수라고 부르고 일인칭은 저를 쓴다."),
        ("preference", "tea", "조용한 찻집과 녹차를 좋아한다."),
        ("procedure", "backup", "매주 백업을 수행하고 저장 위치는 외장 디스크로 한다."),
        ("environment", "shared_server", "사내 공용 서버는 업무 시간에만 운영된다."),
        ("fact", "inventory", "재고 조사는 매월 마지막 금요일에 한다."),
        ("procedure", "deploy", "배포는 금요일을 피하고 화요일 오전에 실행한다."),
    ],
    "unrelated": [
        "회의실 예약 규칙은 어디에 있나요?",
        "노트북 재고는 충분한가요?",
        "신입 사원 교육은 언제 시작하나요?",
        "비품 주문은 누가 승인하나요?",
        "사무실 프린터 토너는 어디에 있나요?",
        "주차장 계약을 갱신하고 싶습니다",
    ],
    "related": [
        ("조용한 찻집을 찾고 있어요", "tea"),
        ("백업 저장 위치는?", "backup"),
        ("공용 서버 운영 시간은?", "shared_server"),
        ("재고 조사는 언제 하나요?", "inventory"),
        ("배포는 언제 하나요?", "deploy"),
    ],
}

CHINESE = {
    "slots": [
        ("identity", "name_pronoun", "把用户称为小李，第一人称使用我。"),
        ("preference", "tea", "喜欢安静的茶馆和绿茶。"),
        ("procedure", "backup", "每周执行备份，保存位置是外接硬盘。"),
        ("environment", "shared_server", "内部共享服务器只在工作时间内运行。"),
        ("fact", "inventory", "盘点在每月最后一个周五进行。"),
        ("procedure", "deploy", "部署避开周五，安排在周二上午。"),
    ],
    "unrelated": [
        "会议室的预订规则在哪里？",
        "办公电脑的库存够用吗？",
        "新员工培训什么时候开始？",
        "办公用品由谁审批？",
        "办公室打印机的碳粉放在哪里？",
        "停车场的合同需要续签",
    ],
    "related": [
        ("想找安静的茶馆", "tea"),
        ("备份保存在哪里？", "backup"),
        ("共享服务器什么时候运行？", "shared_server"),
        ("盘点什么时候进行？", "inventory"),
        ("什么时候部署？", "deploy"),
    ],
}

FIXTURES = {"ja": JAPANESE, "ko": KOREAN, "zh": CHINESE}

CANONICAL_PATHS = (
    ("recall", _canonical_recall_rows),
    ("prefetch", _canonical_prefetch_rows),
)
CANONICAL_PATH_IDS = [name for name, _ in CANONICAL_PATHS]


class FakeCanonicalStore:
    """Minimal canonical store: the gates only need ``list(owner_id)``."""

    def __init__(self, rows):
        self._rows = rows

    def list(self, owner_id):  # noqa: ARG002 - owner scoping is core-provided
        return self._rows


def _store(language):
    return FakeCanonicalStore([
        {
            "body": body,
            "category": category,
            "name": name,
            "created_at": "2026-01-01T00:00:00Z",
        }
        for category, name, body in FIXTURES[language]["slots"]
    ])


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
@pytest.mark.parametrize("language", sorted(FIXTURES))
def test_unrelated_cjk_query_returns_no_canonical_rows(language, path_name, path):
    store = _store(language)

    matched = {
        query: [row.get("canonical_name") for row in path(store, "default", query)]
        for query in FIXTURES[language]["unrelated"]
    }

    assert matched == {query: [] for query in FIXTURES[language]["unrelated"]}, (
        f"{language}/{path_name} injected unrelated canonical slots"
    )


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
@pytest.mark.parametrize("language", sorted(FIXTURES))
def test_cjk_true_positives_are_preserved_and_exclusive(language, path_name, path):
    store = _store(language)
    related = dict(FIXTURES[language]["related"])

    results = {
        query: [row.get("canonical_name") for row in path(store, "default", query)]
        for query in related
    }

    assert results == {query: [name] for query, name in related.items()}, (
        f"{language}/{path_name} did not return exactly the matching slot"
    )


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_single_topical_unit_query_still_matches(path_name, path):
    """Chinese questions often carry one meaningful 2-gram plus function words.

    "什么时候部署？" shares only "部署" with the deploy slot. Function-word units
    must not dilute query coverage into a miss.
    """

    store = _store("zh")

    names = [row.get("canonical_name") for row in path(store, "default", "什么时候部署？")]

    assert "deploy" in names, f"{path_name} dropped a one-topical-unit match: {names}"


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_single_character_cjk_runs_still_match(path_name, path):
    """Comma-separated short CJK entries produce one-character runs."""

    store = FakeCanonicalStore([
        {"body": "猫、犬", "category": "fact", "name": "pets", "created_at": "2026-01-01T00:00:00Z"},
    ])

    names = [row.get("canonical_name") for row in path(store, "default", "猫")]

    assert names == ["pets"]


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_single_character_cjk_run_does_not_match_a_different_character(path_name, path):
    store = FakeCanonicalStore([
        {"body": "猫、犬", "category": "fact", "name": "pets", "created_at": "2026-01-01T00:00:00Z"},
    ])

    assert path(store, "default", "猿") == []


def test_cjk_stop_units_are_operator_configurable(monkeypatch):
    """MNEMOSYNE_PREFETCH_CJK_STOP_UNITS replaces the built-in defaults.

    Setting it to a topical unit proves the knob is live: the Chinese
    "什么时候部署？" question then loses the single unit that made it match.
    """

    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CJK_STOP_UNITS", "什么,么时,时候,部署")

    assert _canonical_recall_rows(_store("zh"), "default", "什么时候部署？") == []


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_latin_canonical_matching_is_unchanged(path_name, path):
    """Latin words keep their pre-#971 behaviour: two shared words still match."""

    store = FakeCanonicalStore([
        {
            "body": "SampleOwner prefers Cedar Bakery over Harbor Bakery.",
            "category": "preference",
            "name": "bakery",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "body": "A lightweight workflow unrelated to tea.",
            "category": "workflow",
            "name": "workflow",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ])

    names = [row.get("canonical_name") for row in path(store, "default", "Cedar Bakery preference")]

    assert "bakery" in names
    assert "workflow" not in names
