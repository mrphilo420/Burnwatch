#!/usr/bin/env python3
"""Stop-slop style-marker detector.

Detects predictable AI writing patterns from the stop-slop catalog
(phrases.md + structures.md): throat-clearing openers, emphasis crutches,
business jargon, filler phrases, meta-commentary, binary contrasts,
passive constructions, and em-dashes.

Larger slop score = more suspicious. The detector is lexical only — it
needs no generative model — and can join the conformal action family as a
prefix-aggregated score.
"""

import re

# Bump when the pattern catalog changes: cached calibration scores were
# computed with an older map and must not be mixed with new scores.
SLOP_VERSION = 3

# (label, regex) — all matched case-insensitively with boundaries where
# phrase anchors exist. Single-word patterns keep word boundaries.
PATTERNS = [
    # throat-clearing openers
    ("here's the thing", r"here['’]s the thing"),
    ("here's what/why/this/that", r"here['’]s (?:what|why|this|that)"),
    ("here's the problem", r"here['’]s the problem"),
    ("uncomfortable truth", r"uncomfortable truth"),
    ("it turns out", r"it turns out"),
    ("let me be clear", r"let me be clear"),
    ("the truth is", r"the truth is"),
    ("i'll say it again", r"i['’]ll say it again"),
    ("i'm going to be honest", r"i['’]m going to be honest"),
    ("can we talk about", r"can we talk about"),
    ("the real X is", r"the real (?:issue|problem|question|reason) is"),
    # emphasis crutches
    ("full stop", r"full stop"),
    ("let that sink in", r"let that sink in"),
    ("this matters because", r"this matters because"),
    ("make no mistake", r"make no mistake"),
    # business jargon
    ("game-changer", r"game[- ]changer"),
    ("double down", r"double down"),
    ("deep dive", r"deep dive"),
    ("take a step back", r"take a step back"),
    ("moving forward", r"moving forward"),
    ("circle back", r"circle back"),
    ("on the same page", r"on the same page"),
    ("delve", r"\bdelve"),
    ("lean into", r"lean into"),
    ("unpack", r"\bunpack(?:ing|ed)?\b"),
    ("navigate the X", r"navigate the (?:complexities|challenges|landscape)"),
    # filler phrases
    ("at its core", r"at its core"),
    ("in today's X", r"in today['’]s"),
    ("it's worth noting", r"it['’]s worth noting|it is worth noting"),
    ("at the end of the day", r"at the end of the day"),
    ("when it comes to", r"when it comes to"),
    ("in a world where", r"in a world where"),
    ("the reality is", r"the reality is"),
    ("in conclusion", r"in conclusion"),
    ("furthermore", r"\bfurthermore"),
    ("moreover", r"\bmoreover"),
    ("additionally", r"\badditionally"),
    ("it's important to note", r"it['’]s important to note|it is important to note"),
    ("it's essential to", r"it['’]s essential to|it is essential to"),
    ("plays a X role", r"plays? a (?:crucial|vital|pivotal|key|significant|important) role"),
    ("wide range of", r"a wide range of"),
    ("myriad", r"\bmyriad"),
    ("plethora", r"\bplethora"),
    ("testament to", r"testament to"),
    ("tapestry", r"\btapestry"),
    ("in the realm of", r"in the realm of"),
    ("in an era of", r"in an era of|in an age of"),
    ("ever-evolving", r"ever[- ]evolving|ever[- ]changing|rapidly evolving"),
    ("unprecedented", r"\bunprecedented"),
    ("transformative", r"\btransformative"),
    ("revolutionary", r"\brevolutionary"),
    ("cutting-edge", r"cutting[- ]edge"),
    ("seamless", r"\bseamless"),
    ("leverage", r"\bleverage"),
    ("streamline", r"\bstreamline"),
    ("elevate", r"\belevate(?:d)?\b"),
    ("foster", r"\bfoster(?:ing)?\b"),
    ("empower", r"\bempower"),
    ("unlock", r"\bunlock(?:ing|ed)?\b"),
    ("embark on", r"embark(?:s|ed)? on"),
    ("harness", r"\bharness(?:ing|es)?\b"),
    ("capitalize on", r"capitali[sz]e on"),
    ("underscore", r"\bunderscore(?:s|d)?\b"),
    ("showcase", r"\bshowcase(?:s|d)?\b"),
    ("pivotal", r"\bpivotal"),
    ("crucial", r"\bcrucial"),
    ("vital", r"\bvital"),
    ("undoubtedly", r"\bundoubtedly"),
    ("without a doubt", r"without a doubt"),
    ("in essence", r"in essence"),
    ("in summary", r"in summary|to summarize"),
    ("all in all", r"all in all"),
    ("as mentioned earlier", r"as mentioned earlier|as previously discussed"),
    ("it should be noted", r"it should be noted|it is worth mentioning"),
    ("keep in mind", r"keep in mind"),
    ("that said", r"that said|having said that"),
    ("with that in mind", r"with that in mind"),
    ("in light of this", r"in light of this"),
    ("it goes without saying", r"it goes without saying|needless to say"),
    ("it is evident that", r"it is evident that|it is clear that|it can be seen that"),
    ("it is widely believed", r"it is widely believed|it is generally accepted"),
    ("the data shows", r"the data (?:shows|tells us)|the findings reveal|the results indicate"),
    ("this underscores", r"this (?:underscores|highlights)"),
    ("the research suggests", r"the research suggests"),
    ("please note", r"please note"),
    ("due to the fact that", r"due to the fact that"),
    ("in other words", r"in other words"),
    ("ultimately", r"\bultimately"),
    ("arguably", r"\barguably"),
    ("importantly", r"\bimportantly"),
    ("notably", r"\bnotably"),
    ("interestingly", r"\binterestingly"),
    ("it's no surprise", r"it['’]s no surprise|it comes as no surprise"),
    ("a beacon of", r"a beacon of"),
    ("stands as a", r"stands as a"),
    ("serves as a", r"serves as a"),
    ("aims to", r"aims to|seeks to|strives to|endeavors to"),
    ("one of the most important", r"one of the most important|one of the key|one of the main"),
    ("competitive edge", r"competitive edge"),
    ("digital landscape", r"digital landscape"),
    ("forward-thinking", r"forward[- ]thinking|forward[- ]looking"),
    # adverb offenders (AI-typical subset of the catalog)
    ("genuinely", r"\bgenuinely"),
    ("fundamentally", r"\bfundamentally"),
    ("inherently", r"\binherently"),
    ("inevitably", r"\binevitably"),
    ("crucially", r"\bcrucially"),
    ("truly", r"\btruly"),
    # structures: binary contrasts and formulaic pivots
    ("not only", r"not only"),
    ("isn't the X", r"isn['’]t (?:the problem|the issue|the answer|just)"),
    ("the answer isn't", r"the answer isn['’]t|the question isn['’]t"),
    ("not X but Y", r"not (?:because|just|about) .{0,60}? but(?: because)? "),
    ("isn't X, it's Y", r"isn['’]t .{1,24},? it['’]s "),
    ("it's not about X, it's about Y", r"it['’]s not (?:just )?about .{0,60}?it['’]s about"),
    ("not because X, because Y", r"not because .{0,60}? because "),
    ("negative-listing striptease", r"it wasn['’]t .{0,30}? it wasn['’]t"),
    ("that's it, that's X", r"that['’]s it[.,] that['’]s"),
    ("that's okay", r"(?:and )?that['’]s ok(?:ay)?\."),
    ("em-dash", r"—"),
    # meta-commentary and rhetorical setups
    ("meta-commentary", r"in this (?:article|essay|blog|post)|this (?:article|essay|blog) (?:will|is)|in the following sections|the rest of this|let me (?:walk|take) you through|we['’]?ll explore|we will explore|as we['’]?ll see|here['’]s what i mean|i want to explore|plot twist|dressed up as|(?:is|are) a feature, not a bug"),
    ("think about it", r"think about it"),
    ("what if", r"what if (?:we|you|i|i told you)"),
    ("that's another post", r"that['’]s another (?:post|article|essay)"),
    # limited passive constructions
    ("passive", r"it is believed that|it was decided|it can be argued|it was created"
                r"|is widely regarded|has been shown to|it has been observed"
                r"|was found to be|is believed to be|was designed to|was developed to"
                r"|is considered to be|were said to"),
    # false agency: inanimate subjects doing human verbs (structures.md)
    ("false agency", r"the market rewards|the market punishes|the culture shifts"
                     r"|the conversation moves toward|the (?:decision|decisions) emerges?"
                     r"|the data tells us|the findings reveal|the results indicate"),
    # vague declaratives (phrases.md: telling instead of showing)
    ("vague declarative", r"the reasons are structural|the implications are significant|the stakes are high|the consequences are real|this is the deepest problem"),
    # performative emphasis / false intimacy
    ("performative", r"\bcreeps? in\b"),
    ("i promise", r"i promise"),
    # telling instead of showing
    ("genuinely hard", r"this is genuinely hard"),
    ("actually looks like", r"this is what .{0,40} actually looks like"),
    ("actually matters", r"actually matters"),
    ("unlocks something", r"unlock(?:s|ed|ing)? something"),
    # classic AI phrases (skill quick checks and example cadence)
    ("cannot be overstated", r"cannot be overstated|can['’]t be overstated"),
    ("has never been more important", r"has never been more important"),
    ("you already know this", r"you already know this"),
    ("let's be honest", r"let['’]s be honest|let['’]s face it|let['’]s be real"),
    ("stating the obvious", r"stating the obvious"),
    ("the bottom line is", r"the bottom line is"),
    ("to put it simply", r"to put it simply|put simply|in simple terms"),
    ("in the grand scheme of things", r"in the grand scheme of things"),
    ("there's no denying", r"there['’]s no denying|there is no denying"),
    ("when all is said and done", r"when all is said and done"),
    ("sheds light on", r"shed(?:s|ding)? light on"),
    ("offers a glimpse into", r"offer(?:s|ing)? a glimpse into"),
    ("underscores the importance", r"underscore(?:s|d)? the importance|highlight(?:s|ed)? the importance"),
    ("it's worth remembering", r"it['’]s worth remembering|it is important to remember"),
    ("in an increasingly X world", r"in an increasingly .{0,24}? world"),
    ("you can only pick two", r"you can only pick two"),
    ("not always, not perfectly", r"not always[.,] not perfectly|not perfectly[.,] not always"),
    ("whether you like it or not", r"whether you like it or not"),
]

# Soft-tier patterns: common in human writing too. Counted separately and
# never used for the conformal score; they are surfaced as weak signals.
SOFT_PATTERNS = [
    ("lazy extreme", r"\b(?:every|always|never|everyone|everybody|nobody)\b"),
    ("hedge adverb", r"\b(?:really|just|actually|simply|deeply|honestly|absolutely|incredibly|literally|very)\b"),
    ("kind of / sort of", r"\bkind of\b|\bsort of\b"),
    ("by the time X, I was Y", r"by the time .{0,40}?,? i (?:was|had)"),
    ("for the most part", r"for the most part"),
    ("by and large", r"by and large"),
    ("not to mention", r"not to mention"),
    ("landscape", r"\blandscape"),
    ("robust", r"\brobust"),
    ("seamless", r"\bseamless"),
    ("ultimately", r"\bultimately"),
    ("navigate", r"\bnavigate\b"),
]

_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in PATTERNS]
_SOFT_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in SOFT_PATTERNS]

# Highlight categories for the inline marked-text view.
_CATEGORY_DEFAULT = "phrase"
CATEGORIES = {
    "opener": {"here's the thing", "here's what/why/this/that", "here's the problem",
               "uncomfortable truth", "it turns out", "let me be clear", "the truth is",
               "i'll say it again", "i'm going to be honest", "can we talk about",
               "the real X is"},
    "filler": {"full stop", "let that sink in", "this matters because", "make no mistake",
               "at its core", "in today's X", "it's worth noting", "at the end of the day",
               "when it comes to", "in a world where", "the reality is", "in conclusion",
               "furthermore", "moreover", "additionally", "it's important to note",
               "it's essential to", "plays a X role", "wide range of", "myriad", "plethora",
               "testament to", "tapestry", "in the realm of", "in an era of", "ever-evolving",
               "unprecedented", "transformative", "revolutionary", "cutting-edge", "seamless",
               "leverage", "streamline", "elevate", "foster", "empower", "unlock", "embark on",
               "harness", "capitalize on", "underscore", "showcase", "pivotal", "crucial",
               "vital", "undoubtedly", "without a doubt", "in essence", "in summary",
               "all in all", "as mentioned earlier", "it should be noted", "keep in mind",
               "that said", "with that in mind", "in light of this", "it goes without saying",
               "it is evident that", "it is widely believed", "the data shows",
               "this underscores", "the research suggests", "please note",
               "due to the fact that", "in other words", "ultimately", "arguably",
               "importantly", "notably", "interestingly", "it's no surprise", "a beacon of",
               "stands as a", "serves as a", "aims to", "one of the most important",
               "competitive edge", "digital landscape", "forward-thinking",
               "genuinely", "fundamentally", "inherently", "inevitably", "crucially", "truly"},
    "jargon": {"game-changer", "double down", "deep dive", "take a step back", "moving forward",
               "circle back", "on the same page", "delve", "lean into", "unpack",
               "navigate the X"},
    "structure": {"not only", "isn't the X", "the answer isn't", "not X but Y",
                  "isn't X, it's Y", "it's not about X, it's about Y", "that's okay",
                  "not because X, because Y", "negative-listing striptease",
                  "that's it, that's X", "vague declarative", "genuinely hard",
                  "actually looks like", "actually matters", "unlocks something",
                  "false agency", "cannot be overstated", "has never been more important",
                  "you already know this", "let's be honest", "stating the obvious",
                  "the bottom line is", "to put it simply",
                  "in the grand scheme of things", "there's no denying",
                  "when all is said and done", "sheds light on", "offers a glimpse into",
                  "underscores the importance", "it's worth remembering",
                  "in an increasingly X world", "you can only pick two",
                  "not always, not perfectly", "whether you like it or not"},
    "meta": {"meta-commentary", "think about it", "what if", "that's another post"},
    "passive": {"passive"},
    "filler": {"performative", "i promise"},
    "emdash": {"em-dash"},
}


def category_of(name):
    for cat, names in CATEGORIES.items():
        if name in names:
            return cat
    return _CATEGORY_DEFAULT


def scan(text):
    """Return marker counts, per-100-words density, example snippets, and
    the exact match spans (for inline highlighting). Hard patterns drive the
    conformal score; soft patterns are reported as weak signals only."""
    words = text.split()
    n_words = max(1, len(words))
    hits = {}
    examples = {}
    spans = []
    for name, rx in _COMPILED:
        ms = list(rx.finditer(text))
        if ms:
            hits[name] = len(ms)
            i = ms[0].start()
            examples[name] = text[max(0, i - 40):min(len(text), i + 80)].replace("\n", " ")
            for m in ms:
                spans.append({"name": name, "category": category_of(name),
                             "start": m.start(), "end": m.end()})
    soft = {}
    soft_examples = {}
    for name, rx in _SOFT_COMPILED:
        ms = list(rx.finditer(text))
        if ms:
            soft[name] = len(ms)
            soft_examples[name] = text[max(0, ms[0].start() - 40):min(len(text), ms[0].start() + 80)].replace("\n", " ")
    total = sum(hits.values())
    soft_total = sum(soft.values())
    return {
        "total": total,
        "density_per_100": round(100.0 * total / n_words, 2),
        "hits": hits,
        "examples": examples,
        "spans": spans,
        "soft_total": soft_total,
        "soft_density_per_100": round(100.0 * soft_total / n_words, 2),
        "soft_hits": soft,
        "soft_examples": soft_examples,
        "n_words": n_words,
    }


def per_word_marks(text):
    """Array over whitespace-separated words: 1 where a pattern overlaps the
    word. Used as a per-token score for prefix aggregation. A word is marked
    only when its exact character span overlaps the match span."""
    words = text.split()
    marks = [0] * len(words)
    if not words:
        return marks
    offsets = []
    ends = []
    pos = 0
    for w in words:
        pos = text.find(w, pos)
        offsets.append(pos)
        pos += len(w)
        ends.append(pos)
    for _, rx in _COMPILED:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            for j, (ws, we) in enumerate(zip(offsets, ends)):
                if ws < e and we > s:
                    marks[j] += 1
    return marks