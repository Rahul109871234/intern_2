"""Mint (livemint.com) news digest -> Telegram (+ optional email), with a PDF of ALL articles.

Daily mode (default): collects the last HOURS (default 24) of articles from Mint's RSS feeds,
  sends a short Telegram message (counts + AI highlights) and the full PDF as a Telegram document.
  If SMTP_USER / SMTP_PASS are set, the PDF is also emailed.
Search mode (QUERY set): searches Mint news for a name/topic (last 7 days) and sends it to Telegram.
"""
import io
import os
import re
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote_plus
from xml.sax.saxutils import escape as xml_escape

import feedparser
import requests
from bs4 import BeautifulSoup

# Most specific categories first, so an article is filed under its specific section.
FEEDS = {
    "Markets": "https://www.livemint.com/rss/markets",
    "Companies": "https://www.livemint.com/rss/companies",
    "Money": "https://www.livemint.com/rss/money",
    "Industry": "https://www.livemint.com/rss/industry",
    "Technology": "https://www.livemint.com/rss/technology",
    "AI": "https://www.livemint.com/rss/AI",
    "Politics": "https://www.livemint.com/rss/politics",
    "Elections": "https://www.livemint.com/rss/elections",
    "Sports": "https://www.livemint.com/rss/sports",
    "Science": "https://www.livemint.com/rss/science",
    "Education": "https://www.livemint.com/rss/education",
    "Insurance": "https://www.livemint.com/rss/insurance",
    "Opinion": "https://www.livemint.com/rss/opinion",
    "News": "https://www.livemint.com/rss/news",
}

HOURS = int(os.environ.get("HOURS") or 24)
MAX_PER_CATEGORY = 40
IST = timezone(timedelta(hours=5, minutes=30))
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL") or "qwen3-32b"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HIGHLIGHTS_PROMPT = ("You are given today's business news headlines from an Indian newspaper. Write "
                     "5 short bullet points (start each with '- ') of the most important things "
                     "happening, using ONLY the headlines. Do not invent facts. No preamble, no thinking.")
OVERVIEW_PROMPT = ("You are given news headlines about one person or topic. Write a 3 sentence "
                   "plain-text overview of what is being reported, using ONLY the headlines. "
                   "Do not invent facts. No preamble, no thinking.")


# ---------- LLM (free Groq) ----------
def llm(system: str, text: str, fallback: str = "") -> str:
    if not GROQ_API_KEY:
        return fallback
    for model in (GROQ_MODEL, GROQ_FALLBACK_MODEL):
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": model, "temperature": 0.2,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": text}]},
                timeout=60,
            )
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            out = re.sub(r"<think>.*?(</think>|$)", "", out, flags=re.S).strip()
            if out:
                return out
        except Exception as e:
            print(f"LLM call with {model} failed: {e}")
    return fallback


# ---------- Fetching ----------
def clean(markup: str) -> str:
    return BeautifulSoup(markup or "", "html.parser").get_text(" ", strip=True)


def excerpt(text: str, limit: int = 350) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def read_feed(url: str):
    r = requests.get(url, headers=UA, timeout=25)
    r.raise_for_status()
    return feedparser.parse(r.content)


def search_news(query: str, days: int, limit: int, category: str = "Search"):
    """Google News RSS search (used for name search, and as a fallback if Mint's feeds are blocked)."""
    q = quote_plus(f"{query} when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = read_feed(url)
    except Exception as e:
        print(f"Search failed for {query!r}: {e}")
        return []
    items, seen = [], set()
    for e in feed.entries:
        link, title = e.get("link"), e.get("title", "")
        if not link or title in seen:
            continue
        seen.add(title)
        ts = e.get("published_parsed")
        published = datetime(*ts[:6], tzinfo=timezone.utc) if ts else datetime.now(timezone.utc)
        items.append({"category": category, "title": title, "link": link,
                      "published": published, "desc": ""})
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:limit]


def fetch_mint():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS)
    seen, items, failures = set(), [], 0
    for category, url in FEEDS.items():
        try:
            feed = read_feed(url)
        except Exception as e:
            failures += 1
            print(f"Feed failed {category}: {e}")
            continue
        count = 0
        for e in feed.entries:
            link = e.get("link")
            ts = e.get("published_parsed") or e.get("updated_parsed")
            title = (e.get("title") or "").strip()
            if not link or not ts or not title or link in seen:
                continue
            published = datetime(*ts[:6], tzinfo=timezone.utc)
            if published < cutoff or count >= MAX_PER_CATEGORY:
                continue
            seen.add(link)
            count += 1
            items.append({"category": category, "title": title, "link": link, "published": published,
                          "desc": excerpt(clean(e.get("summary", "")))})
    if not items and failures:
        print("All Mint feeds failed or empty; falling back to Google News (site:livemint.com).")
        items = search_news("site:livemint.com", days=max(1, HOURS // 24), limit=100,
                            category="Mint (via Google News)")
    items.sort(key=lambda x: x["published"], reverse=True)
    return items


def group_by_category(items):
    order = list(FEEDS) + ["Mint (via Google News)", "Search"]
    groups = {}
    for i in items:
        groups.setdefault(i["category"], []).append(i)
    return [(c, groups[c]) for c in sorted(groups, key=lambda c: order.index(c) if c in order else 99)]


# ---------- PDF ----------
def build_pdf(items, heading: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    regular, bold, unicode_ok = "Helvetica", "Helvetica-Bold", False
    for r_path, b_path in [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ]:
        if os.path.exists(r_path) and os.path.exists(b_path):
            pdfmetrics.registerFont(TTFont("DV", r_path))
            pdfmetrics.registerFont(TTFont("DV-B", b_path))
            pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")
            regular, bold, unicode_ok = "DV", "DV-B", True
            break

    def t(s):  # text -> safe for the chosen font, XML-escaped
        s = s or ""
        if not unicode_ok:
            s = s.replace("₹", "Rs ").encode("latin-1", "replace").decode("latin-1")
        return xml_escape(s)

    base = getSampleStyleSheet()["Normal"]
    h1 = ParagraphStyle("h1", parent=base, fontName=bold, fontSize=18, leading=22, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=base, fontName=regular, fontSize=9, textColor="#666666", spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=base, fontName=bold, fontSize=13, leading=16, spaceBefore=12,
                        spaceAfter=6, textColor="#0b5d3b")
    title_st = ParagraphStyle("title", parent=base, fontName=bold, fontSize=10, leading=13)
    meta = ParagraphStyle("meta", parent=base, fontName=regular, fontSize=8, textColor="#777777", leading=10)
    body = ParagraphStyle("body", parent=base, fontName=regular, fontSize=9, leading=12, spaceAfter=7)

    story = [Paragraph(t(heading), h1),
             Paragraph(t(f"{len(items)} articles | generated {datetime.now(IST).strftime('%d %b %Y, %I:%M %p')} IST"), sub)]
    for category, group in group_by_category(items):
        story.append(Paragraph(t(f"{category} ({len(group)})"), h2))
        for i in group:
            url = xml_escape(i["link"], {'"': "&quot;"})
            story.append(Paragraph(f'<a href="{url}" color="#1a4fa0">{t(i["title"])}</a>', title_st))
            story.append(Paragraph(t(i["published"].astimezone(IST).strftime("%d %b, %I:%M %p")), meta))
            story.append(Paragraph(t(i["desc"]) if i.get("desc") else "&nbsp;", body))
    if not items:
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("No new articles found.", body))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=15 * mm, bottomMargin=15 * mm, title=heading).build(story)
    return buf.getvalue()


# ---------- Delivery ----------
def tg_call(method: str, **kwargs):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    r = requests.post(f"https://api.telegram.org/bot{token}/{method}", timeout=120, **kwargs)
    if not r.ok:  # never print the URL (it contains the bot token)
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:300]}")


def send_text(messages):
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    for text in messages:
        tg_call("sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                                     "disable_web_page_preview": True})
        time.sleep(1)


def send_pdf_telegram(pdf: bytes, filename: str, caption: str):
    tg_call("sendDocument", data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "caption": caption[:1000]},
            files={"document": (filename, pdf, "application/pdf")})


def send_pdf_email(pdf: bytes, filename: str, subject: str):
    user, password = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not (user and password):
        return
    to = os.environ.get("EMAIL_TO") or user
    msg = MIMEMultipart()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.attach(MIMEText("The Mint news digest is attached as a PDF.", "plain"))
    part = MIMEApplication(pdf, _subtype="pdf")
    part.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(part)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, password)
        s.sendmail(user, [x.strip() for x in to.split(",")], msg.as_string())
    print("PDF emailed")


def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chunk(blocks, limit=3800):
    out, cur = [], ""
    for b in blocks:
        if len(cur) + len(b) + 1 > limit and cur:
            out.append(cur)
            cur = ""
        cur += b + "\n"
    if cur.strip():
        out.append(cur)
    return out


# ---------- Modes ----------
def run_daily():
    date = datetime.now(IST).strftime("%d %b %Y")
    items = fetch_mint()
    if not items:
        send_text([f"<b>Mint News - {date}</b>\nNo new articles found in the last {HOURS} hours."])
        print("No articles")
        return
    groups = group_by_category(items)
    counts = ", ".join(f"{c} {len(g)}" for c, g in groups)
    headlines = "\n".join(f"- {i['title']}" for i in items[:60])
    highlights = llm(HIGHLIGHTS_PROMPT, headlines)

    text = f"<b>Mint News - {date}</b>\n{len(items)} articles: {esc(counts)}\n"
    if highlights:
        text += f"\n<b>Highlights</b>\n{esc(highlights)}\n"
    else:
        top = "\n".join(f"- {esc(i['title'])}" for i in items[:8])
        text += f"\n<b>Latest</b>\n{top}\n"
    send_text([text])

    filename = f"mint_news_{datetime.now(IST).strftime('%Y-%m-%d')}.pdf"
    pdf = build_pdf(items, f"Mint News - {date}")
    send_pdf_telegram(pdf, filename, f"Mint News - {date}: full list of {len(items)} articles")
    send_pdf_email(pdf, filename, f"Mint News - {date}")
    print(f"Sent {len(items)} articles")


def run_search(query: str):
    items = search_news(f"{query} site:livemint.com", days=7, limit=10)
    if not items:
        send_text([f"No recent Mint news found for \"{esc(query)}\"."])
        return
    overview = llm(OVERVIEW_PROMPT, f"Topic: {query}\n" + "\n".join(f"- {i['title']}" for i in items))
    blocks = [f"<b>Mint news: {esc(query)}</b> (last 7 days)\n"]
    if overview:
        blocks.append(f"<i>{esc(overview)}</i>\n")
    for n, i in enumerate(items, 1):
        link = i["link"].replace("&", "&amp;").replace('"', "&quot;")
        blocks.append(f'{n}. <a href="{link}">{esc(i["title"])}</a>\n')
    send_text(chunk(blocks))
    print(f"Sent search results for {query!r}")


if __name__ == "__main__":
    q = (os.environ.get("QUERY") or "").strip()
    if q:
        run_search(q[:100])
    else:
        run_daily()
