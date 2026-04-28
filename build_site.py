#!/usr/bin/env python3
"""Parse README.md and generate a polished static website."""

import re, html as H

def md_to_html(text):
    """Convert markdown links/bold/italic to HTML, escaping the rest."""
    out = ''
    i = 0
    while i < len(text):
        if text[i] == '[':
            m = re.match(r'\[([^\]]*)\]\(([^)]*)\)', text[i:])
            if m:
                label = H.escape(m.group(1))
                href = m.group(2)
                out += f'<a href="{H.escape(href)}" target="_blank" rel="noopener">{label}</a>'
                i += m.end()
                continue
        out += H.escape(text[i])
        i += 1
    return out


def split_table_row(line):
    """Split a markdown table row by | while respecting []() links."""
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    cells = []
    depth_sq = 0
    depth_rd = 0
    current = ''
    for ch in line:
        if ch == '[':
            depth_sq += 1
        elif ch == ']':
            depth_sq = max(0, depth_sq - 1)
        elif ch == '(':
            depth_rd += 1
        elif ch == ')':
            depth_rd = max(0, depth_rd - 1)
        if ch == '|' and depth_sq == 0 and depth_rd == 0:
            cells.append(current.strip())
            current = ''
        else:
            current += ch
    cells.append(current.strip())
    return cells


def is_separator_row(cells):
    return all(re.match(r'^[-:\s]*$', c) for c in cells if c)


def parse_table_block(lines):
    """Parse table lines into header + data rows."""
    all_rows = []
    for line in lines:
        cells = split_table_row(line)
        cells = [c for c in cells if c.strip() != '' or len(cells) > 2]
        if cells:
            all_rows.append(cells)

    if len(all_rows) < 2:
        return None, []

    header = all_rows[0]
    data = []
    for row in all_rows[1:]:
        if is_separator_row(row):
            continue
        data.append(row)
    return header, data


def render_table(header, data):
    if not header:
        return ''
    ncols = len(header)
    h = '<div class="table-wrap"><table><thead><tr>'
    for c in header:
        h += f'<th>{md_to_html(c)}</th>'
    h += '</tr></thead><tbody>'
    for row in data:
        h += '<tr>'
        for i in range(ncols):
            cell = row[i] if i < len(row) else ''
            h += f'<td>{md_to_html(cell)}</td>'
        h += '</tr>'
    h += '</tbody></table></div>'
    return h


def is_table_line(line):
    stripped = line.strip()
    if stripped.startswith('|'):
        return True
    pipe_count = stripped.count('|')
    if pipe_count >= 3 and not stripped.startswith('#') and not stripped.startswith('```'):
        return True
    return False


def parse_readme(path):
    with open(path, 'r') as f:
        content = f.read()

    sections = []
    current_sec = None
    current_sub = None
    table_lines = []
    in_table = False
    in_code = False

    def flush_table():
        nonlocal table_lines, in_table
        if table_lines:
            target = current_sub if current_sub else current_sec
            if target:
                target['blocks'].append({'type': 'table', 'lines': list(table_lines)})
            table_lines = []
        in_table = False

    for line in content.split('\n'):
        stripped = line.strip()

        if stripped.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue

        is_heading = re.match(r'^(#{2,6})\s+', stripped)
        if is_heading:
            flush_table()
            level = len(is_heading.group(1))
            title = stripped[is_heading.end():]
            title = re.sub(r'<a\s+name=[\'"][^\'"]*[\'"]></a>', '', title).strip()
            title = re.sub(r'^[\d.]+\s*', '', title).strip()
            title = re.sub(r'#####?\s*', '', title).strip()

            if level == 2:
                current_sub = None
                current_sec = {'title': title, 'level': 2, 'blocks': [], 'subs': []}
                sections.append(current_sec)
            elif current_sec:
                current_sub = {'title': title, 'level': level, 'blocks': []}
                current_sec['subs'].append(current_sub)
            continue

        if is_table_line(stripped):
            in_table = True
            table_lines.append(stripped)
        else:
            if in_table:
                if stripped == '' and table_lines:
                    continue
                flush_table()

    flush_table()
    return sections


SECTION_META = [
    ('vlm',         'SoTA VLMs',                'fa-robot',      '#6366f1', 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&q=80'),
    ('bench',       'Benchmarks',               'fa-chart-bar',  '#0ea5e9', 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&q=80'),
    ('post',        'Post-Training',            'fa-fire',       '#f59e0b', 'https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800&q=80'),
    ('app',         'Applications',             'fa-cogs',       '#10b981', 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&q=80'),
    ('challenge',   'Challenges',               'fa-shield-alt', '#ef4444', 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=800&q=80'),
]

def get_meta(title):
    tl = title.lower()
    for key, _, icon, color, img in SECTION_META:
        if key in tl or _ .lower() in tl:
            return icon, color, img
    return 'fa-book', '#8b5cf6', 'https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&q=80'


def count_entries(sec):
    n = 0
    for b in sec.get('blocks', []):
        if b['type'] == 'table':
            header, data = parse_table_block(b['lines'])
            n += len(data)
    for sub in sec.get('subs', []):
        n += count_entries(sub)
    return n


def build_html(sections):
    skip = {
        'table of contents',
        'citation',
        'the evolution of vlm architectures',
        "what's in this repo",
    }
    filtered = [s for s in sections if s['title'].lower().strip('📄🗂️⚒️⛑️📚🔥🧭🆕 ') not in skip
                and not s['title'].lower().startswith('citation')]
    filtered = [s for s in filtered if s['title'].strip()]

    nav = ''
    body = ''

    for i, sec in enumerate(filtered):
        icon, color, img = get_meta(sec['title'])
        sid = f's{i}'
        count = count_entries(sec)
        badge = f'<span class="badge" style="background:{color}22;color:{color}">{count}</span>' if count else ''
        nav += f'<a href="#{sid}" class="nav-link" data-color="{color}"><i class="fas {icon} nav-icon" style="color:{color}"></i>{H.escape(sec["title"])}{badge}</a>\n'

        sub_nav = ''
        sub_html = ''
        for j, sub in enumerate(sec.get('subs', [])):
            sub_id = f'{sid}-{j}'
            sub_count = count_entries(sub)
            sub_badge = f'<span class="badge-sm">{sub_count}</span>' if sub_count else ''
            sub_nav += f'<a href="#{sub_id}" class="nav-sub">{H.escape(sub["title"])} {sub_badge}</a>\n'

            tables_html = ''
            for b in sub.get('blocks', []):
                if b['type'] == 'table':
                    header, data = parse_table_block(b['lines'])
                    if header:
                        tables_html += render_table(header, data)
            if tables_html:
                sub_html += f'''<div class="subsection" id="{sub_id}">
<button class="sub-toggle" onclick="toggleSection(this)" aria-expanded="false">
<span>{md_to_html(sub["title"])} {sub_badge}</span>
<svg class="chevron" width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M5 7.5L10 12.5L15 7.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
</button>
<div class="sub-content">{tables_html}</div>
</div>\n'''

        nav += sub_nav

        sec_tables = ''
        for b in sec.get('blocks', []):
            if b['type'] == 'table':
                header, data = parse_table_block(b['lines'])
                if header:
                    sec_tables += render_table(header, data)

        body += f'''<section class="section" id="{sid}">
<div class="section-banner" style="--accent:{color}">
<img src="{img}" alt="" class="banner-img" loading="lazy"/>
<div class="banner-overlay"></div>
<div class="banner-text">
<i class="fas {icon} banner-icon" style="color:{color}"></i>
<h2>{sec["title"]}</h2>
<p class="entry-count">{count} entries</p>
</div>
</div>
<div class="section-content">
{sec_tables}
{sub_html}
</div>
</section>\n'''

    stat_counts = {}
    for sec in filtered:
        tl = sec['title'].lower()
        n = count_entries(sec)
        if 'vlm' in tl and 'sota' in tl.replace('ö','o'):
            stat_counts['models'] = n
        elif 'benchmark' in tl or 'evaluation' in tl:
            stat_counts['benchmarks'] = n
        elif 'post' in tl and 'training' in tl:
            stat_counts['training'] = n
        elif 'application' in tl:
            stat_counts['apps'] = n

    html = HTML_TEMPLATE.replace('{{NAV}}', nav).replace('{{BODY}}', body)
    for key, val in stat_counts.items():
        html = html.replace(f'id="stat-{key}">--<', f'id="stat-{key}">{val}<')
    return html


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Vision-Language Models &mdash; Survey Overview</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:#0b1120;--bg2:#111827;--surface:#1a2332;--surface2:#243044;
  --border:rgba(255,255,255,.06);--text:#e8ecf4;--text2:#8896ab;--text3:#5a6a80;
  --radius:10px;--radius-lg:16px;
  --accent:#818cf8;--accent2:#6366f1;
  --transition:all .25s cubic-bezier(.4,0,.2,1);
}

html{scroll-behavior:smooth;scroll-padding-top:80px}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none;transition:color .2s}
a:hover{color:#a5b4fc}

/* ─── HERO ─── */
.hero{position:relative;min-height:55vh;display:flex;align-items:center;justify-content:center;text-align:center;overflow:hidden}
.hero-bg{position:absolute;inset:0}
.hero-bg::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 30% 20%,rgba(99,102,241,.15),transparent 60%),radial-gradient(ellipse at 70% 80%,rgba(6,182,212,.1),transparent 50%),var(--bg)}
.hero-bg::after{content:'';position:absolute;inset:0;background:url('https://images.unsplash.com/photo-1676299081847-824916de030a?w=1400&q=75') center/cover;opacity:.08}
.hero-content{position:relative;z-index:2;max-width:820px;padding:3rem 1.5rem}
.hero h1{font-size:clamp(1.8rem,5vw,3.2rem);font-weight:900;letter-spacing:-.02em;background:linear-gradient(135deg,#818cf8 0%,#06b6d4 50%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.6rem;line-height:1.15}
.hero .subtitle{font-size:1.05rem;color:var(--text2);max-width:560px;margin:0 auto 2rem;line-height:1.7}
.pills{display:flex;gap:.6rem;justify-content:center;flex-wrap:wrap;margin-bottom:2rem}
.pill{padding:.4rem 1rem;border-radius:99px;font-size:.75rem;font-weight:600;background:rgba(255,255,255,.04);border:1px solid var(--border);color:var(--text2);display:inline-flex;align-items:center;gap:.4rem}
.pill i{font-size:.65rem}
.stats{display:flex;gap:1rem;justify-content:center;flex-wrap:wrap}
.stat-card{background:rgba(255,255,255,.03);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:var(--radius);padding:.8rem 1.4rem;min-width:110px;text-align:center;transition:var(--transition)}
.stat-card:hover{border-color:rgba(129,140,248,.3);transform:translateY(-2px)}
.stat-num{font-size:1.6rem;font-weight:800;color:var(--accent);display:block}
.stat-label{font-size:.68rem;color:var(--text3);text-transform:uppercase;letter-spacing:.06em;margin-top:.1rem}
.hero-links{display:flex;gap:.8rem;margin-top:1.8rem;justify-content:center;flex-wrap:wrap}
.cite-link{display:inline-flex;align-items:center;gap:.5rem;padding:.6rem 1.4rem;background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);border-radius:99px;color:var(--accent);font-size:.85rem;font-weight:600;transition:var(--transition)}
.cite-link:hover{background:rgba(99,102,241,.2);text-decoration:none;transform:translateY(-1px)}
.cite-link.github-link{background:rgba(255,255,255,.06);border-color:rgba(255,255,255,.15);color:var(--text)}
.cite-link.github-link:hover{background:rgba(255,255,255,.12)}

/* ─── SEARCH BAR ─── */
.search-wrap{position:sticky;top:0;z-index:50;background:rgba(11,17,32,.85);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);padding:.6rem 1.5rem}
.search-inner{max-width:1400px;margin:0 auto;display:flex;align-items:center;gap:.8rem}
.search-box{flex:1;display:flex;align-items:center;background:var(--surface);border:1px solid var(--border);border-radius:99px;padding:0 1rem;transition:var(--transition)}
.search-box:focus-within{border-color:rgba(129,140,248,.4);box-shadow:0 0 0 3px rgba(129,140,248,.1)}
.search-box i{color:var(--text3);font-size:.85rem}
.search-box input{flex:1;background:none;border:none;color:var(--text);padding:.55rem .7rem;font-size:.85rem;font-family:inherit;outline:none}
.search-box input::placeholder{color:var(--text3)}
.search-results-info{font-size:.75rem;color:var(--text3);white-space:nowrap}
.expand-all-btn{padding:.45rem 1rem;border-radius:99px;border:1px solid var(--border);background:var(--surface);color:var(--text2);font-size:.78rem;font-family:inherit;cursor:pointer;transition:var(--transition);white-space:nowrap}
.expand-all-btn:hover{border-color:rgba(129,140,248,.3);color:var(--text)}

/* ─── LAYOUT ─── */
.layout{display:flex;max-width:1400px;margin:0 auto}
.sidebar{position:sticky;top:50px;align-self:flex-start;height:calc(100vh - 50px);width:270px;min-width:270px;background:var(--bg2);border-right:1px solid var(--border);overflow-y:auto;padding:1.2rem 0;scrollbar-width:thin;scrollbar-color:var(--surface2) transparent}
.sidebar::-webkit-scrollbar{width:4px}
.sidebar::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:4px}
.sidebar-title{padding:.2rem 1.2rem .8rem;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3)}
.nav-link{display:flex;align-items:center;gap:.6rem;padding:.5rem 1.2rem;font-size:.8rem;color:var(--text2);border-left:3px solid transparent;transition:var(--transition);overflow:hidden}
.nav-link:hover,.nav-link.active{color:var(--text);background:rgba(255,255,255,.03);text-decoration:none}
.nav-link.active{border-left-color:var(--accent)}
.nav-icon{width:16px;text-align:center;font-size:.75rem;flex-shrink:0}
.nav-sub{display:block;padding:.3rem 1.2rem .3rem 2.8rem;font-size:.73rem;color:var(--text3);transition:var(--transition)}
.nav-sub:hover{color:var(--text2);text-decoration:none}
.badge{font-size:.65rem;padding:.1rem .45rem;border-radius:99px;margin-left:auto;font-weight:600;flex-shrink:0}
.badge-sm{font-size:.65rem;color:var(--text3);opacity:.7}

/* ─── MAIN ─── */
.main{flex:1;min-width:0}
.section{margin-bottom:1px}
.section-banner{position:relative;height:170px;overflow:hidden;display:flex;align-items:flex-end}
.banner-img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;filter:brightness(.35) saturate(1.3);transition:filter .5s}
.section:hover .banner-img{filter:brightness(.42) saturate(1.4)}
.banner-overlay{position:absolute;inset:0;background:linear-gradient(0deg,var(--bg) 0%,transparent 70%)}
.banner-text{position:relative;z-index:2;padding:1.2rem 2rem;width:100%}
.banner-icon{font-size:1.3rem;margin-bottom:.25rem;display:block;filter:drop-shadow(0 0 8px currentColor)}
.banner-text h2{font-size:1.4rem;font-weight:700;letter-spacing:-.01em}
.entry-count{font-size:.75rem;color:var(--text3);margin-top:.15rem}

.section-content{padding:1.2rem 2rem 2rem}

/* ─── TABLE ─── */
.table-wrap{overflow-x:auto;margin-bottom:1.2rem;border-radius:var(--radius);border:1px solid var(--border);background:var(--surface)}
table{width:100%;border-collapse:collapse;font-size:.8rem}
thead{position:sticky;top:0;z-index:1}
th{padding:.65rem 1rem;text-align:left;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--text2);background:var(--surface2);border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:.55rem 1rem;border-bottom:1px solid rgba(255,255,255,.03);color:var(--text2);max-width:320px;vertical-align:top}
td a{font-weight:500}
tbody tr{transition:background .15s}
tbody tr:hover td{background:rgba(255,255,255,.02);color:var(--text)}
tbody tr.hidden{display:none}

/* ─── SUBSECTIONS ─── */
.subsection{margin-bottom:.6rem;border-radius:var(--radius);border:1px solid var(--border);overflow:hidden;background:var(--surface);transition:border-color .2s}
.subsection:hover{border-color:rgba(255,255,255,.1)}
.sub-toggle{width:100%;display:flex;justify-content:space-between;align-items:center;padding:.75rem 1.1rem;background:none;border:none;color:var(--text);font-size:.88rem;font-weight:600;cursor:pointer;font-family:inherit;text-align:left;transition:background .15s}
.sub-toggle:hover{background:rgba(255,255,255,.02)}
.chevron{transition:transform .3s;color:var(--text3);flex-shrink:0}
.sub-toggle[aria-expanded="true"] .chevron{transform:rotate(180deg)}
.sub-content{max-height:0;overflow:hidden;transition:max-height .45s cubic-bezier(.4,0,.2,1),padding .3s}
.sub-content.open{max-height:12000px;padding:.2rem 1rem 1rem}

/* ─── BACK TO TOP ─── */
.back-top{position:fixed;bottom:1.5rem;right:1.5rem;width:42px;height:42px;border-radius:50%;background:rgba(99,102,241,.15);backdrop-filter:blur(10px);border:1px solid rgba(99,102,241,.3);color:var(--accent);display:flex;align-items:center;justify-content:center;cursor:pointer;opacity:0;pointer-events:none;transition:var(--transition);z-index:100;font-size:1rem}
.back-top.show{opacity:1;pointer-events:all}
.back-top:hover{background:rgba(99,102,241,.25);transform:translateY(-2px)}

/* ─── FOOTER ─── */
.footer{text-align:center;padding:2.5rem 2rem;color:var(--text3);font-size:.8rem;border-top:1px solid var(--border)}
.footer a{color:var(--text2)}

/* ─── RESPONSIVE ─── */
@media(max-width:960px){
  .sidebar{display:none}
  .section-banner{height:130px}
  .section-content{padding:1rem}
  .banner-text{padding:1rem}
  td,th{padding:.4rem .6rem;font-size:.72rem}
  .hero{min-height:45vh}
  .hero h1{font-size:1.6rem}
}
@media(max-width:600px){
  .search-inner{flex-wrap:wrap}
  .expand-all-btn{display:none}
  .stats{gap:.5rem}
  .stat-card{padding:.6rem .8rem;min-width:80px}
  .stat-num{font-size:1.2rem}
}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="hero-bg"></div>
  <div class="hero-content">
    <div class="pills">
      <span class="pill"><i class="fas fa-circle"></i>CVPR 2025 Workshop</span>
      <span class="pill"><i class="fas fa-circle"></i>Updated Apr 2026</span>
      <span class="pill"><i class="fas fa-circle"></i>Open Source</span>
    </div>
    <h1>Vision-Language Models<br/>Survey &amp; Overview</h1>
    <p class="subtitle">A curated collection of state-of-the-art VLMs, benchmarks, RL alignment methods, applications, and open challenges in the multimodal AI landscape.</p>
    <div class="stats">
      <div class="stat-card"><span class="stat-num" id="stat-models">--</span><span class="stat-label">Models</span></div>
      <div class="stat-card"><span class="stat-num" id="stat-benchmarks">--</span><span class="stat-label">Benchmarks</span></div>
      <div class="stat-card"><span class="stat-num" id="stat-training">--</span><span class="stat-label">RL &amp; SFT</span></div>
      <div class="stat-card"><span class="stat-num" id="stat-apps">--</span><span class="stat-label">Applications</span></div>
    </div>
    <div class="hero-links">
      <a class="cite-link" href="https://arxiv.org/abs/2501.02189" target="_blank"><i class="fas fa-file-alt"></i>Read the Paper</a>
      <a class="cite-link github-link" href="https://github.com/zli12321/Vision-Language-Models-Overview" target="_blank"><i class="fab fa-github"></i>GitHub</a>
    </div>
  </div>
</div>

<!-- SEARCH -->
<div class="search-wrap">
  <div class="search-inner">
    <div class="search-box">
      <i class="fas fa-search"></i>
      <input type="text" id="searchInput" placeholder="Search models, papers, benchmarks..." autocomplete="off"/>
    </div>
    <span class="search-results-info" id="searchInfo"></span>
    <button class="expand-all-btn" id="expandBtn" onclick="toggleAll()">Expand All</button>
  </div>
</div>

<!-- LAYOUT -->
<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-title">Contents</div>
    {{NAV}}
  </nav>
  <main class="main">
    {{BODY}}
  </main>
</div>

<!-- FOOTER -->
<div class="footer">
  <p>Built from the <a href="https://arxiv.org/abs/2501.02189" target="_blank">VLM Survey</a> &mdash; Li et al., CVPR 2025 Workshop &mdash; Last updated April 2026</p>
</div>

<button class="back-top" id="backTop" onclick="window.scrollTo({top:0,behavior:'smooth'})"><i class="fas fa-arrow-up"></i></button>

<script>
/* Toggle subsection */
function toggleSection(btn){
  const expanded = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', !expanded);
  btn.nextElementSibling.classList.toggle('open');
}

/* Expand / Collapse all */
let allExpanded = false;
function toggleAll(){
  allExpanded = !allExpanded;
  document.querySelectorAll('.sub-toggle').forEach(b => {
    b.setAttribute('aria-expanded', allExpanded);
    b.nextElementSibling.classList.toggle('open', allExpanded);
  });
  document.getElementById('expandBtn').textContent = allExpanded ? 'Collapse All' : 'Expand All';
}

/* Search */
const searchInput = document.getElementById('searchInput');
const searchInfo = document.getElementById('searchInfo');
let debounce;
searchInput.addEventListener('input', () => {
  clearTimeout(debounce);
  debounce = setTimeout(doSearch, 200);
});
function doSearch(){
  const q = searchInput.value.trim().toLowerCase();
  const rows = document.querySelectorAll('tbody tr');
  let shown = 0, total = rows.length;
  rows.forEach(r => {
    if(!q){
      r.classList.remove('hidden');
      shown++;
    } else {
      const text = r.textContent.toLowerCase();
      const match = text.includes(q);
      r.classList.toggle('hidden', !match);
      if(match) shown++;
    }
  });
  if(q){
    searchInfo.textContent = `${shown} of ${total} entries`;
    /* auto-expand subsections with matches */
    document.querySelectorAll('.subsection').forEach(sub => {
      const vis = sub.querySelectorAll('tbody tr:not(.hidden)').length;
      if(vis > 0){
        const btn = sub.querySelector('.sub-toggle');
        btn.setAttribute('aria-expanded','true');
        sub.querySelector('.sub-content').classList.add('open');
      }
    });
  } else {
    searchInfo.textContent = '';
  }
}

/* Active nav */
const navLinks = document.querySelectorAll('.nav-link');
const sectionEls = document.querySelectorAll('.section');
const backTop = document.getElementById('backTop');
window.addEventListener('scroll', () => {
  backTop.classList.toggle('show', window.scrollY > 400);
  let current = '';
  sectionEls.forEach(s => {
    if(window.scrollY >= s.offsetTop - 200) current = s.id;
  });
  navLinks.forEach(l => {
    const isActive = l.getAttribute('href') === '#' + current;
    l.classList.toggle('active', isActive);
    if(isActive) l.style.borderLeftColor = l.dataset.color;
    else l.style.borderLeftColor = 'transparent';
  });
}, {passive: true});
</script>
</body>
</html>'''


if __name__ == '__main__':
    sections = parse_readme('README.md')
    html = build_html(sections)
    with open('index.html', 'w') as f:
        f.write(html)
    row_count = html.count('<tr>')
    print(f'Built index.html — {len(sections)} sections, ~{row_count} table rows')
