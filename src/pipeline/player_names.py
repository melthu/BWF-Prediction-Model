"""
Canonical player names.

Wikipedia spells the same player several ways - different word order
("Kidambi Srikanth" / "Srikanth Kidambi"), optional name parts ("Anthony
Ginting" / "Anthony Sinisuka Ginting"), inconsistent hyphenation and case,
and dropped diacritics. Left alone, each spelling becomes a separate player:
its own Elo, its own form, its own head-to-head record. Parupalli Kashyap's
history was split almost down the middle, and Prannoy's four ways.

This map is deliberately explicit rather than a normalisation rule applied at
runtime. Two real players can have names that normalise to the same string,
and silently merging them is a far worse error than leaving one player split.
Every entry below was checked against three pieces of evidence: same
nationality, plausible career span, and - decisively - whether the two names
ever appear in the same draw or play each other.

Deliberately NOT merged, despite looking alike:

* "Huang Yu" / "Huang Yu-kai" - they played each other in the third round of
  Kaohsiung Masters 2023. Different people.
* "Munawar Mohammed" (India, 2018) / "Mohammed Munawar" (UAE, 2025).

The Mauritian "Georges Paul" / "Julien Paul" / "Georges Julien Paul" trio read
the same way and were held back until confirmed; they are one player and are
merged below.

Note that sharing a draw is not on its own disqualifying: a player can advance
through a bracket under two spellings when Wikipedia's own page is
inconsistent (Parupalli Kashyap, China Open 2012; Arnaud Merklé, Syed Modi
2023). Only actually meeting is conclusive.

`data_checks.py` reports any new collision that is not covered here, so a
future spelling gets reviewed rather than silently splitting a player again.
"""

import unicodedata

# alias -> canonical. Canonical is the most frequent spelling, breaking ties
# toward diacritics, then the shorter and less capitalised form.
#
# Vietnamese names are the exception: they are rendered without diacritics
# throughout, matching how BWF and Wikipedia's English pages write them. That
# keeps "Nguyen Tien Minh" (which won on frequency) and "Nguyen Hai Dang"
# (which would otherwise have won its tie on the diacritic rule) consistent
# with each other.
ALIASES = {
    "Chou Tien-Chen": "Chou Tien-chen",
    "Chen Chou-tien": "Chou Tien-chen",
    "Chou Tien Chen": "Chou Tien-chen",
    "Ng Ka Long Angus": "Ng Ka Long",
    "Angus Ng Ka Long": "Ng Ka Long",
    "Kidambi Srikanth": "Srikanth Kidambi",
    "Anthony Ginting": "Anthony Sinisuka Ginting",
    "Wang Tzu-Wei": "Wang Tzu-wei",
    "H. S. Prannoy": "Prannoy H. S.",
    "H.S. Prannoy": "Prannoy H. S.",
    "Prannoy H.S.": "Prannoy H. S.",
    "Sai Praneeth B.": "B. Sai Praneeth",
    "Kashyap Parupalli": "Parupalli Kashyap",
    "Daren Liew": "Liew Daren",
    "Lee Dong Keun": "Lee Dong-keun",
    "Thammasin Sitthikom": "Sitthikom Thammasin",
    "Manjunath Mithun": "Mithun Manjunath",
    "Dionysius Rumbaka": "Dionysius Hayom Rumbaka",
    "Nguyễn Tiến Minh": "Nguyen Tien Minh",
    "Arnaud Merkle": "Arnaud Merklé",
    "Shon Wan Ho": "Shon Wan-ho",
    "S.Sankar Muthusamy Subramanian": "Sankar Subramanian",
    "Pablo Abian": "Pablo Abián",
    "Ygor Coelho de Oliveira": "Ygor Coelho",
    "Lucas Corvee": "Lucas Corvée",
    "Aidil Sholeh Ali Sadikin": "Aidil Sholeh",
    "Huan Gao": "Gao Huan",
    "Song Xue": "Xue Song",
    "Sathish Kumar Karunakaran": "Sathish Karunakaran",
    "Sheng Xiaodong": "Xiaodong Sheng",
    "Hsueh Hsuan-Yi": "Hsueh Hsuan-yi",
    "Kartikey Kumar": "Kartikey Gulshan Kumar",
    "Meiraba Luwang Maisnam": "Meiraba Maisnam",
    "Saputra Vicky Angga": "Vicky Angga Saputra",
    "Jason Ho-Shue": "Jason Ho-shue",
    "Christian Lind": "Christian Lind Thomsen",
    "Chun Seang Tan": "Tan Chun Seang",
    "Bismo Oktora": "Bismo Raya Oktora",
    "Hsieh Yu Hsing": "Hsieh Yu-hsing",
    "Rahul Bharadwaj": "B. M. Rahul Bharadwaj",
    "Rahul Bharadwaj B.M": "B. M. Rahul Bharadwaj",
    "Zi Liang Derek Wong": "Derek Wong",
    "Derek Wong Zi Liang": "Derek Wong",
    "Kestutis Navickas": "Kęstutis Navickas",
    "Ashton Chen Yong Zhao": "Yong Zhao Ashton Chen",
    "Lakshay SHARMA": "Lakshay Sharma",
    "Rohan Kumar": "A R Rohan Kumar",
    "Chun Kar Lung": "Kar Lung Chun",
    "Kavin Thangam Kavin": "Kavin Thangam",
    "Thangam Kavin": "Kavin Thangam",
    "Ansh Gupta": "Ansh Vishal Gupta",
    "Hemanth M. Gowda": "Hemanth Gowda",
    "Ville Lang": "Ville Lång",
    "Yuhan Tan": "Tan Yuhan",
    "M Atef Haikal Taufik": "M. Atef Haikal Taufik",
    "Chan Jie Ying": "Jie Ying Chan",
    "Nguyễn Hai Dang": "Nguyen Hai Dang",
    "Kho Henrikho Wibowo": "Henrikho Kho Wibowo",
    "Henrikho Wibowo Kho": "Henrikho Kho Wibowo",
    # Both found by the bracket rewiring in build_bracket, which could not
    # match a draw slot to the player who had just won into it. Neither pair
    # has ever met; each is one player under a dominant spelling and a rare one
    # (25 matches vs 2, 19 vs 1), and each appears both ways inside a single
    # draw - Hyderabad Open 2018 and Orleans Masters 2026 respectively.
    "R. M. V. Gurusaidutt": "Gurusai Dutt",
    "Dong Tian Yao": "Dong Tianyao",
    "Joel Koh": "Joel Koh Jia Wei",
    "Ryan Ng Zin Rei": "Ryan Ng",
    "K. Ajay Kumar": "Ajay Kumar K.",
    "Hsieh Yu-Hsin": "Hsieh Yu-hsin",
    # Both India, overlapping span (2022-24), never met, and "Ravi Ravi" appears
    # only within one draw. A doubled single-token name is how Wikipedia renders
    # a mononym in a table that wants two fields, so this is one player.
    "Ravi Ravi": "Ravi",
    # Confirmed one player. Unlike the other merges this one keeps the *fullest*
    # spelling rather than the most frequent: the counts are 5/4/1, so there is
    # no dominant rendering to defer to, and picking between two truncations on
    # a one-match margin would be arbitrary.
    "Georges Paul": "Georges Julien Paul",
    "Julien Paul": "Georges Julien Paul",

    # Two more full-name spellings, both confirmed on the player's own
    # Wikipedia article rather than assumed: "Luis Montoya (badminton)" opens
    # "Luis Armando Montoya Navarro is a Mexican badminton player", and
    # "Vladimir Malkov (badminton)" opens "Vladimir Vadimovich Malkov is a
    # Russian badminton player". Worth the check - unlike a dropped accent,
    # a dropped name part is exactly how two different people collide.
    "Luis Armando Montoya Navarro": "Luis Montoya",
    "Vladimir Vadimovich Malkov": "Vladimir Malkov",

    # Vietnamese names are rendered without diacritics throughout, matching
    # BWF and the English Wikipedia. Folding "Nguyễn Hải Đăng" also reunites
    # it with the "Nguyen Hai Dang" spelling it had been split from.
    "Bùi Thành Đạt": "Bui Thanh Dat",
    "Lê Đức Phát": "Le Duc Phat",
    # The World Championships pages accent these names only half-way -
    # "Lê Đuc Phat" keeps the stroke on the D but drops the tone marks, and
    # "Nguyen Hai Đang" keeps only the stroke. Neither matches the fully
    # accented key above, so each arrived as a brand-new player.
    "Lê Đuc Phat": "Le Duc Phat",
    "Nguyễn Hoàng Nam": "Nguyen Hoang Nam",
    "Nguyễn Hải Đăng": "Nguyen Hai Dang",
    # Same name again, but the 2026 Vietnam Open page spells the D as U+00D0
    # ETH rather than U+0110 D-WITH-STROKE. The two are visually identical and
    # fold_ascii maps both to "D", but ALIASES is an exact dict lookup, so the
    # line above did not match and the player split in two all over again.
    "Nguyễn Hải Ðăng": "Nguyen Hai Dang",
    "Nguyen Hai Đang": "Nguyen Hai Dang",
    "Nguyễn Thu Thảo": "Nguyen Thu Thao",
    "Nguyễn Tiến Tuấn": "Nguyen Tien Tuan",
    "Nguyễn Văn Mai": "Nguyen Van Mai",
    "Nguyễn Đình Hoàng": "Nguyen Dinh Hoang",
    "Phan Phúc Thịnh": "Phan Phuc Thinh",
    "Phạm Cao Cường": "Pham Cao Cuong",
    "Trần Lê Mạnh An": "Tran Le Manh An",
    "Trần Quốc Khánh": "Tran Quoc Khanh",

    # Diacritic variants elsewhere fold toward the correct spelling, which
    # is also the majority one in each case.
    "Adrian Dziolko": "Adrian Dziółko",
    "Ditlev Jaeger Holm": "Ditlev Jæger Holm",
    "Jan O. Jorgensen": "Jan Ø. Jørgensen",
    "Milan Ludik": "Milan Ludík",
    "Michal Rogalski": "Michał Rogalski",
    "Przemyslaw Wacha": "Przemysław Wacha",
}


# Pairs that look like variants but have been checked and judged distinct (or
# too ambiguous to merge). Recorded so data_checks stops asking: a warning that
# fires on every run is a warning nobody reads.
REVIEWED_DISTINCT = {
    frozenset(("Huang Yu", "Huang Yu-kai")),
    frozenset(("Munawar Mohammed", "Mohammed Munawar")),
}


# Letters formed with a stroke or slash are single codepoints, not a base plus
# a combining mark, so NFKD leaves them intact and the ASCII pass then drops
# them outright: "Đăng" folds to "ang", "Mikołaj" to "mikoaj". That silently
# hid a split player and produced mangled URL slugs, so fold them explicitly.
_STROKE_LETTERS = str.maketrans({
    "Đ": "D", "đ": "d", "Ð": "D", "ð": "d",
    "Ł": "L", "ł": "l",
    "Ø": "O", "ø": "o",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "Þ": "Th", "þ": "th",
    "ß": "ss",
    "Ħ": "H", "ħ": "h",
    "Ŧ": "T", "ŧ": "t",
    "Ɖ": "D", "Ƶ": "Z", "ƶ": "z",
})


def fold_ascii(name: str) -> str:
    """Diacritic-free ASCII form of a name, for comparison and URL slugs."""
    return (unicodedata.normalize("NFKD", name.translate(_STROKE_LETTERS))
            .encode("ascii", "ignore").decode())


def canonical(name):
    """Map one player name to its canonical spelling. Passes through unknowns."""
    if not isinstance(name, str):
        return name
    return ALIASES.get(name.strip(), name.strip())


def canonicalise(df):
    """Rewrite every player column of a match frame in place and return it."""
    for col in ("player_a", "player_b"):
        if col in df.columns:
            df[col] = df[col].map(canonical)
    return df
