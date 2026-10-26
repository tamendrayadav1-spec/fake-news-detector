"""
generate_dataset.py
--------------------
Builds a labeled training dataset (data/news_dataset.csv) for the
Fake News Detection System.

NOTE FOR STUDENTS (viva point):
This project ships with a synthetically constructed dataset so the
whole pipeline runs offline, out-of-the-box, without needing to
download a large external corpus. It is built from realistic
templates of how REAL (factual, attributed, measured) news and FAKE
(sensational, unverified, emotionally manipulative) news are
typically written, combined with topic/entity variation to create
a reasonably diverse set of examples.

For a stronger final submission, you can drop in a larger public
dataset such as the Kaggle "Fake and Real News Dataset" (has
Fake.csv / True.csv) into the data/ folder, rename columns to
`text` and `label` (1 = FAKE, 0 = REAL), and re-run train_model.py.
The rest of the pipeline (Flask app, explainability, credibility
scoring) needs no changes.
"""

import csv
import random

random.seed(42)

# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------

TOPICS = [
    "the economy", "a new vaccine", "the national election", "climate policy",
    "a celebrity", "a tech company", "the stock market", "a cricket match",
    "a government scheme", "an educational policy", "a health study",
    "a viral social media trend", "a natural disaster", "a court verdict",
    "a new smartphone launch", "a state budget", "an army operation",
    "a space mission", "a banking rule", "a food safety issue",
    "a university exam", "a railway announcement", "a farmers' protest",
    "an international summit", "a currency change", "a religious event",
]

ENTITIES = [
    "the Ministry of Health", "Reserve Bank of India", "the Prime Minister's Office",
    "a local hospital", "the Election Commission", "NASA", "ISRO",
    "the World Health Organization", "a Supreme Court bench", "the Finance Ministry",
    "a state government", "the Railway Board", "a university administration",
    "the Reserve Bank", "a district collector's office", "the Central Government",
    "a leading research institute", "the United Nations", "a state police department",
]

REAL_TEMPLATES = [
    "According to a statement released by {entity} on Monday, {topic} will be reviewed "
    "after officials completed a preliminary assessment. The report, based on data "
    "collected over the past quarter, notes that further details will be shared "
    "once the analysis is peer-reviewed.",

    "{entity} confirmed in a press briefing that {topic} is progressing as scheduled. "
    "Officials cited figures from an internal audit and said the next update would be "
    "issued after the committee meets later this month.",

    "A report published by {entity} states that {topic} showed a moderate change "
    "compared to last year, based on verified figures from multiple independent "
    "sources. Experts interviewed for the report urged caution against over-interpreting "
    "short-term data.",

    "In an official notification, {entity} clarified the current status of {topic}, "
    "adding that the decision followed consultations with stakeholders and was reviewed "
    "by a designated committee before being made public.",

    "Data released by {entity} on {topic} was cross-checked with independent analysts, "
    "who broadly agreed with the reported figures, though they noted minor gaps that "
    "require further study before firm conclusions can be drawn.",

    "{entity} issued a clarification regarding {topic}, stating that earlier reports "
    "circulating online were only partially accurate and that the full findings would "
    "be published in an upcoming bulletin available on the official website.",

    "Officials from {entity} told reporters that {topic} is being monitored closely, "
    "with a formal review expected within the next few weeks. No final decision has "
    "been announced yet, they said.",

    "A study conducted in coordination with {entity} examined {topic} using a sample "
    "collected over several months. The findings were shared at a press conference and "
    "are available for public review.",
]

FAKE_TEMPLATES = [
    "SHOCKING!!! You won't BELIEVE what {entity} is hiding about {topic} — "
    "insiders reveal the TRUTH they don't want you to know!!! Share before this "
    "gets DELETED forever!!!",

    "BREAKING: Secret leaked documents PROVE that {topic} is a complete HOAX "
    "orchestrated by {entity}. Mainstream media REFUSES to report this! Forward "
    "to everyone NOW before it's banned!",

    "Doctors and officials are STUNNED after a whistleblower from {entity} exposed "
    "the REAL agenda behind {topic}. This one trick they don't want revealed will "
    "change everything you thought you knew!",

    "URGENT WARNING: {entity} is secretly planning something about {topic} that will "
    "shock the entire nation. Experts are too scared to speak, but this message must "
    "go viral RIGHT NOW!!!",

    "Everyone is talking about how {entity} LIED about {topic} for years! A retired "
    "official finally speaks out and the truth will leave you SPEECHLESS. Click here "
    "before this video gets taken down!",

    "You've been FOOLED! {topic} was never what {entity} claimed. Anonymous sources "
    "confirm a massive cover-up that the government doesn't want you to see. Wake up "
    "and share this message with 10 friends immediately!",

    "EXPOSED: The hidden truth about {topic} that {entity} tried to bury! This forwarded "
    "message contains information that 'they' don't want going public. Don't let this "
    "disappear — forward NOW!",

    "Miracle claim about {topic} leaves scientists baffled! {entity} allegedly covered "
    "it up for decades, according to an unverified WhatsApp forward now going viral "
    "across social media.",
]

REAL_SOURCES = [
    "pib.gov.in", "reuters.com", "thehindu.com", "bbc.com", "who.int",
    "rbi.org.in", "aajtak.in", "indianexpress.com", "ndtv.com", "livemint.com",
    "timesofindia.indiatimes.com", "isro.gov.in", "prsindia.org",
]

FAKE_SOURCES = [
    "truth4all-news.blogspot.com", "viral-updates24.info", "sharewithall.net",
    "realnewsnowbreaking.xyz", "forward-this-now.co", "secretexposed.click",
    "whatsapp-forward.today", "unbelievablefacts247.win", "newsflash-alert.top",
]


def make_row(is_fake: bool):
    topic = random.choice(TOPICS)
    entity = random.choice(ENTITIES)
    template = random.choice(FAKE_TEMPLATES if is_fake else REAL_TEMPLATES)
    text = template.format(topic=topic, entity=entity)
    source = random.choice(FAKE_SOURCES if is_fake else REAL_SOURCES)
    return {
        "text": text,
        "source": source,
        "label": 1 if is_fake else 0,  # 1 = FAKE, 0 = REAL
    }


def main(n_per_class: int = 350):
    rows = []
    for _ in range(n_per_class):
        rows.append(make_row(is_fake=True))
        rows.append(make_row(is_fake=False))
    random.shuffle(rows)

    out_path = "data/news_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Fake: {sum(r['label'] for r in rows)}  Real: {len(rows) - sum(r['label'] for r in rows)}")


if __name__ == "__main__":
    main()
