#!/usr/bin/env python3
"""
Generate a standalone HTML viewer for reviewing screenshots + labels.

Creates a single HTML file with React that displays images and their
corresponding JSON labels side-by-side. Navigate with arrow keys.

Usage:
    python scripts/generate_review.py
    python scripts/generate_review.py --images data/raw --labels data/labeled
    python scripts/generate_review.py --embed  # Embed images as base64 (no server needed)

Then open the generated HTML file:
    # If using --embed:
    open review.html

    # If not embedding (default, smaller file):
    cd /path/to/chimera && python -m http.server 8000
    # Then open http://localhost:8000/review.html
"""

import argparse
import base64
import json
from pathlib import Path


def image_to_data_uri(image_path: Path) -> str:
    """Convert image to base64 data URI."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{media_type};base64,{data}"


def find_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching image-label pairs."""
    pairs = []
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        label_path = labels_dir / f"{image_path.stem}.json"
        if label_path.exists():
            pairs.append((image_path, label_path))

    return sorted(pairs, key=lambda x: x[0].name)


def find_comparison_data(
    images_dir: Path, claude_dir: Path, predictions_dir: Path
) -> list[dict]:
    """Find matching image, Claude label, and Qwen prediction for comparison."""
    comparisons = []
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        claude_path = claude_dir / f"{image_path.stem}.json"
        # Predictions use _analysis.json suffix
        pred_path = predictions_dir / f"{image_path.stem}_analysis.json"

        if claude_path.exists():
            comparisons.append({
                "image": image_path,
                "claude": claude_path,
                "qwen": pred_path if pred_path.exists() else None
            })

    return sorted(comparisons, key=lambda x: x["image"].name)


def generate_comparison_html(comparisons: list[dict], embed_images: bool = False) -> str:
    """Generate HTML for comparing Claude vs Qwen predictions."""

    # Build data array
    items = []
    for comp in comparisons:
        image_path = comp["image"]
        claude_path = comp["claude"]
        qwen_path = comp["qwen"]

        with open(claude_path) as f:
            claude_data = json.load(f)

        qwen_data = None
        if qwen_path and qwen_path.exists():
            with open(qwen_path) as f:
                qwen_data = json.load(f)

        if embed_images:
            image_src = image_to_data_uri(image_path)
        else:
            image_src = str(image_path)

        items.append({
            "image": image_src,
            "claude": claude_data,
            "qwen": qwen_data,
            "filename": image_path.name
        })

    data_json = json.dumps(items)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chimera - Claude vs Qwen Comparison</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 12px;
            gap: 12px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 16px;
            background: #1a1a1a;
            border-radius: 8px;
        }}
        .nav-info {{
            font-size: 14px;
            color: #888;
        }}
        .filename {{
            font-family: monospace;
            font-size: 14px;
            color: #4fc3f7;
        }}
        .nav-buttons {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .nav-btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            background: #2a2a2a;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .nav-btn:hover {{
            background: #3a3a3a;
        }}
        .nav-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        .jump-input {{
            width: 60px;
            padding: 8px;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            background: #2a2a2a;
            color: #e0e0e0;
            font-size: 14px;
            text-align: center;
        }}
        .jump-input:focus {{
            outline: none;
            border-color: #4fc3f7;
        }}
        .main {{
            display: flex;
            gap: 12px;
            flex: 1;
            min-height: 0;
        }}
        .image-panel {{
            flex: 0 0 40%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #111;
            border-radius: 8px;
            overflow: hidden;
        }}
        .image-panel img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .comparison-panel {{
            flex: 1;
            display: flex;
            gap: 12px;
            min-height: 0;
        }}
        .result-column {{
            flex: 1;
            background: #1a1a1a;
            border-radius: 8px;
            overflow: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
        }}
        .column-header {{
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 8px 12px;
            margin: -16px -16px 12px -16px;
            border-radius: 8px 8px 0 0;
        }}
        .claude-header {{
            background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
            color: white;
        }}
        .qwen-header {{
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            color: white;
        }}
        .no-data {{
            color: #666;
            font-style: italic;
            text-align: center;
            padding: 40px;
        }}
        .json-section {{
            margin-bottom: 16px;
        }}
        .json-section:last-child {{
            margin-bottom: 0;
        }}
        .section-title {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2a2a;
        }}
        .field {{
            display: flex;
            padding: 3px 0;
            font-size: 12px;
        }}
        .field-name {{
            min-width: 110px;
            color: #666;
        }}
        .field-value {{
            color: #e0e0e0;
            word-break: break-word;
        }}
        .field-value.number {{
            color: #81c784;
        }}
        .field-value.null {{
            color: #555;
            font-style: italic;
        }}
        .field-value.match {{
            color: #4ade80;
        }}
        .field-value.mismatch {{
            color: #f87171;
            font-weight: 500;
        }}
        .list-value {{
            color: #ffb74d;
        }}
        .text-block {{
            font-size: 12px;
            line-height: 1.4;
            color: #e0e0e0;
            padding: 6px;
            background: #222;
            border-radius: 4px;
            margin-top: 4px;
        }}
        .help {{
            font-size: 12px;
            color: #666;
        }}
        .stats {{
            font-size: 12px;
            color: #888;
            padding: 4px 8px;
            background: #222;
            border-radius: 4px;
        }}
        .stats .match-count {{
            color: #4ade80;
        }}
        .stats .mismatch-count {{
            color: #f87171;
        }}
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const DATA = {data_json};

        function App() {{
            const [index, setIndex] = React.useState(0);
            const [jumpValue, setJumpValue] = React.useState('');

            const item = DATA[index];
            const total = DATA.length;

            React.useEffect(() => {{
                const handleKey = (e) => {{
                    if (e.target.tagName === 'INPUT') return;
                    if (e.key === 'ArrowLeft' && index > 0) setIndex(i => i - 1);
                    if (e.key === 'ArrowRight' && index < total - 1) setIndex(i => i + 1);
                }};
                window.addEventListener('keydown', handleKey);
                return () => window.removeEventListener('keydown', handleKey);
            }}, [index, total]);

            const handleJump = (e) => {{
                if (e.key === 'Enter') {{
                    const page = parseInt(jumpValue, 10);
                    if (page >= 1 && page <= total) {{
                        setIndex(page - 1);
                        setJumpValue('');
                        e.target.blur();
                    }}
                }}
            }};

            // Compare values and count matches
            const compareFields = (claude, qwen) => {{
                if (!qwen) return {{ matches: 0, total: 0 }};
                const gs_fields = ['map_name', 'round_phase', 'player_side', 'player_health', 'player_armor',
                    'player_money', 'weapon_primary', 'weapon_secondary', 'alive_teammates', 'alive_enemies',
                    'bomb_status', 'visible_enemies'];
                let matches = 0;
                let total = 0;
                const cgs = claude?.game_state || {{}};
                const qgs = qwen?.game_state || {{}};
                gs_fields.forEach(f => {{
                    total++;
                    const cv = cgs[f];
                    const qv = qgs[f];
                    if (cv === qv || (cv === null && qv === null) ||
                        (typeof cv === 'number' && typeof qv === 'number' && cv === qv)) {{
                        matches++;
                    }}
                }});
                return {{ matches, total }};
            }};

            const stats = compareFields(item.claude, item.qwen);

            const renderValue = (val, compareVal = undefined) => {{
                const hasCompare = compareVal !== undefined;
                let matchClass = '';
                if (hasCompare && item.qwen) {{
                    const isMatch = val === compareVal ||
                        (val === null && compareVal === null) ||
                        (typeof val === 'number' && typeof compareVal === 'number' && val === compareVal);
                    matchClass = isMatch ? ' match' : ' mismatch';
                }}

                if (val === null || val === undefined) return <span className={{"field-value null" + matchClass}}>null</span>;
                if (typeof val === 'number') return <span className={{"field-value number" + matchClass}}>{{val}}</span>;
                if (Array.isArray(val)) {{
                    if (val.length === 0) return <span className={{"field-value null" + matchClass}}>[]</span>;
                    return <span className={{"field-value list-value" + matchClass}}>[{{val.join(', ')}}]</span>;
                }}
                return <span className={{"field-value" + matchClass}}>{{String(val)}}</span>;
            }};

            const renderColumn = (data, type, otherData) => {{
                if (!data) {{
                    return <div className="no-data">No {{type}} prediction available</div>;
                }}

                const gs = data.game_state || {{}};
                const ogs = otherData?.game_state || {{}};
                const an = data.analysis || {{}};
                const ad = data.advice || {{}};
                const showComparison = type === 'Qwen' && otherData;

                return (
                    <>
                        <div className="json-section">
                            <div className="section-title">Game State</div>
                            <div className="field"><span className="field-name">Map</span>{{renderValue(gs.map_name, showComparison ? ogs.map_name : undefined)}}</div>
                            <div className="field"><span className="field-name">Phase</span>{{renderValue(gs.round_phase, showComparison ? ogs.round_phase : undefined)}}</div>
                            <div className="field"><span className="field-name">Side</span>{{renderValue(gs.player_side, showComparison ? ogs.player_side : undefined)}}</div>
                            <div className="field"><span className="field-name">Health</span>{{renderValue(gs.player_health, showComparison ? ogs.player_health : undefined)}}</div>
                            <div className="field"><span className="field-name">Armor</span>{{renderValue(gs.player_armor, showComparison ? ogs.player_armor : undefined)}}</div>
                            <div className="field"><span className="field-name">Money</span>{{renderValue(gs.player_money, showComparison ? ogs.player_money : undefined)}}</div>
                            <div className="field"><span className="field-name">Primary</span>{{renderValue(gs.weapon_primary, showComparison ? ogs.weapon_primary : undefined)}}</div>
                            <div className="field"><span className="field-name">Secondary</span>{{renderValue(gs.weapon_secondary, showComparison ? ogs.weapon_secondary : undefined)}}</div>
                            <div className="field"><span className="field-name">Utility</span>{{renderValue(gs.utility)}}</div>
                            <div className="field"><span className="field-name">Teammates</span>{{renderValue(gs.alive_teammates, showComparison ? ogs.alive_teammates : undefined)}}</div>
                            <div className="field"><span className="field-name">Enemies</span>{{renderValue(gs.alive_enemies, showComparison ? ogs.alive_enemies : undefined)}}</div>
                            <div className="field"><span className="field-name">Bomb</span>{{renderValue(gs.bomb_status, showComparison ? ogs.bomb_status : undefined)}}</div>
                            <div className="field"><span className="field-name">Visible</span>{{renderValue(gs.visible_enemies, showComparison ? ogs.visible_enemies : undefined)}}</div>
                        </div>

                        <div className="json-section">
                            <div className="section-title">Analysis</div>
                            <div className="text-block">{{an.situation_summary || 'N/A'}}</div>
                            <div className="field"><span className="field-name">Economy</span>{{renderValue(an.economy_assessment)}}</div>
                            <div className="field"><span className="field-name">Importance</span>{{renderValue(an.round_importance)}}</div>
                        </div>

                        <div className="json-section">
                            <div className="section-title">Advice</div>
                            <div className="text-block">{{ad.primary_action || 'N/A'}}</div>
                        </div>
                    </>
                );
            }};

            return (
                <div className="container">
                    <div className="header">
                        <div>
                            <span className="filename">{{item.filename}}</span>
                            <span className="nav-info"> — {{index + 1}} / {{total}}</span>
                            {{item.qwen && (
                                <span className="stats" style={{{{marginLeft: '12px'}}}}>
                                    Accuracy: <span className="match-count">{{stats.matches}}</span>/<span>{{stats.total}}</span>
                                    {{' '}}({{Math.round(stats.matches / stats.total * 100)}}%)
                                </span>
                            )}}
                        </div>
                        <div className="nav-buttons">
                            <span className="help">← → navigate</span>
                            <button className="nav-btn" onClick={{() => setIndex(i => i - 1)}} disabled={{index === 0}}>← Prev</button>
                            <input
                                type="text"
                                className="jump-input"
                                placeholder={{`1-${{total}}`}}
                                value={{jumpValue}}
                                onChange={{(e) => setJumpValue(e.target.value)}}
                                onKeyDown={{handleJump}}
                            />
                            <button className="nav-btn" onClick={{() => setIndex(i => i + 1)}} disabled={{index === total - 1}}>Next →</button>
                        </div>
                    </div>
                    <div className="main">
                        <div className="image-panel">
                            <img src={{item.image}} alt={{item.filename}} />
                        </div>
                        <div className="comparison-panel">
                            <div className="result-column">
                                <div className="column-header claude-header">Claude (Ground Truth)</div>
                                {{renderColumn(item.claude, 'Claude', null)}}
                            </div>
                            <div className="result-column">
                                <div className="column-header qwen-header">Qwen3.5-27B</div>
                                {{renderColumn(item.qwen, 'Qwen', item.claude)}}
                            </div>
                        </div>
                    </div>
                </div>
            );
        }}

        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>'''

    return html


def generate_html(pairs: list[tuple[Path, Path]], embed_images: bool = False) -> str:
    """Generate the review HTML."""

    # Build data array
    items = []
    for image_path, label_path in pairs:
        with open(label_path) as f:
            label_data = json.load(f)

        if embed_images:
            image_src = image_to_data_uri(image_path)
        else:
            image_src = str(image_path)

        items.append({
            "image": image_src,
            "label": label_data,
            "filename": image_path.name
        })

    data_json = json.dumps(items)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chimera - Screenshot Review</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 12px;
            gap: 12px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 16px;
            background: #1a1a1a;
            border-radius: 8px;
        }}
        .nav-info {{
            font-size: 14px;
            color: #888;
        }}
        .filename {{
            font-family: monospace;
            font-size: 14px;
            color: #4fc3f7;
        }}
        .nav-buttons {{
            display: flex;
            gap: 8px;
        }}
        .nav-btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            background: #2a2a2a;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .nav-btn:hover {{
            background: #3a3a3a;
        }}
        .nav-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        .jump-input {{
            width: 60px;
            padding: 8px;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            background: #2a2a2a;
            color: #e0e0e0;
            font-size: 14px;
            text-align: center;
        }}
        .jump-input:focus {{
            outline: none;
            border-color: #4fc3f7;
        }}
        .main {{
            display: flex;
            gap: 12px;
            flex: 1;
            min-height: 0;
        }}
        .image-panel {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #111;
            border-radius: 8px;
            overflow: hidden;
        }}
        .image-panel img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .json-panel {{
            flex: 1;
            background: #1a1a1a;
            border-radius: 8px;
            overflow: auto;
            padding: 16px;
        }}
        .json-section {{
            margin-bottom: 20px;
        }}
        .json-section:last-child {{
            margin-bottom: 0;
        }}
        .section-title {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #4fc3f7;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2a2a;
        }}
        .field {{
            display: flex;
            padding: 4px 0;
            font-size: 13px;
        }}
        .field-name {{
            min-width: 140px;
            color: #888;
        }}
        .field-value {{
            color: #e0e0e0;
            word-break: break-word;
        }}
        .field-value.number {{
            color: #81c784;
        }}
        .field-value.null {{
            color: #666;
            font-style: italic;
        }}
        .list-value {{
            color: #ffb74d;
        }}
        .text-block {{
            font-size: 13px;
            line-height: 1.5;
            color: #e0e0e0;
            padding: 8px;
            background: #222;
            border-radius: 4px;
            margin-top: 4px;
        }}
        .threat-item, .opportunity-item {{
            padding: 4px 8px;
            margin: 4px 0;
            border-radius: 4px;
            font-size: 13px;
        }}
        .threat-item {{
            background: rgba(244, 67, 54, 0.15);
            border-left: 3px solid #f44336;
        }}
        .opportunity-item {{
            background: rgba(76, 175, 80, 0.15);
            border-left: 3px solid #4caf50;
        }}
        .help {{
            font-size: 12px;
            color: #666;
        }}
        .flag-btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .flag-btn.unflagged {{
            background: #2a2a2a;
            color: #e0e0e0;
        }}
        .flag-btn.flagged {{
            background: #f44336;
            color: white;
        }}
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const DATA = {data_json};

        function App() {{
            const [index, setIndex] = React.useState(0);
            const [jumpValue, setJumpValue] = React.useState('');
            const [flagged, setFlagged] = React.useState(() => {{
                const saved = localStorage.getItem('chimera-flagged');
                return saved ? JSON.parse(saved) : {{}};
            }});

            const item = DATA[index];
            const total = DATA.length;

            React.useEffect(() => {{
                const handleKey = (e) => {{
                    if (e.key === 'ArrowLeft' && index > 0) setIndex(i => i - 1);
                    if (e.key === 'ArrowRight' && index < total - 1) setIndex(i => i + 1);
                    if (e.key === 'f') toggleFlag();
                }};
                window.addEventListener('keydown', handleKey);
                return () => window.removeEventListener('keydown', handleKey);
            }}, [index, total]);

            const toggleFlag = () => {{
                const newFlagged = {{ ...flagged }};
                if (newFlagged[item.filename]) {{
                    delete newFlagged[item.filename];
                }} else {{
                    newFlagged[item.filename] = true;
                }}
                setFlagged(newFlagged);
                localStorage.setItem('chimera-flagged', JSON.stringify(newFlagged));
            }};

            const handleJump = (e) => {{
                if (e.key === 'Enter') {{
                    const page = parseInt(jumpValue, 10);
                    if (page >= 1 && page <= total) {{
                        setIndex(page - 1);
                        setJumpValue('');
                        e.target.blur();
                    }}
                }}
            }};

            const exportFlagged = () => {{
                const list = Object.keys(flagged).join('\\n');
                const blob = new Blob([list], {{ type: 'text/plain' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'flagged.txt';
                a.click();
            }};

            const renderValue = (val) => {{
                if (val === null || val === undefined) return <span className="field-value null">null</span>;
                if (typeof val === 'number') return <span className="field-value number">{{val}}</span>;
                if (Array.isArray(val)) {{
                    if (val.length === 0) return <span className="field-value null">[]</span>;
                    return <span className="field-value list-value">[{{val.join(', ')}}]</span>;
                }}
                return <span className="field-value">{{String(val)}}</span>;
            }};

            const gs = item.label.game_state || {{}};
            const an = item.label.analysis || {{}};
            const ad = item.label.advice || {{}};

            return (
                <div className="container">
                    <div className="header">
                        <div>
                            <span className="filename">{{item.filename}}</span>
                            <span className="nav-info"> — {{index + 1}} / {{total}}</span>
                        </div>
                        <div className="nav-buttons">
                            <span className="help">← → navigate | F flag | Enter to jump</span>
                            <button className="nav-btn" onClick={{() => setIndex(i => i - 1)}} disabled={{index === 0}}>← Prev</button>
                            <input
                                type="text"
                                className="jump-input"
                                placeholder={{`1-${{total}}`}}
                                value={{jumpValue}}
                                onChange={{(e) => setJumpValue(e.target.value)}}
                                onKeyDown={{handleJump}}
                            />
                            <button className="nav-btn" onClick={{() => setIndex(i => i + 1)}} disabled={{index === total - 1}}>Next →</button>
                            <button className={{`flag-btn ${{flagged[item.filename] ? 'flagged' : 'unflagged'}}`}} onClick={{toggleFlag}}>
                                {{flagged[item.filename] ? '⚑ Flagged' : '⚐ Flag'}}
                            </button>
                            {{Object.keys(flagged).length > 0 && (
                                <button className="nav-btn" onClick={{exportFlagged}}>Export ({{Object.keys(flagged).length}})</button>
                            )}}
                        </div>
                    </div>
                    <div className="main">
                        <div className="image-panel">
                            <img src={{item.image}} alt={{item.filename}} />
                        </div>
                        <div className="json-panel">
                            <div className="json-section">
                                <div className="section-title">Game State</div>
                                <div className="field"><span className="field-name">Map</span>{{renderValue(gs.map_name)}}</div>
                                <div className="field"><span className="field-name">Phase</span>{{renderValue(gs.round_phase)}}</div>
                                <div className="field"><span className="field-name">Side</span>{{renderValue(gs.player_side)}}</div>
                                <div className="field"><span className="field-name">Health / Armor</span>{{renderValue(gs.player_health)}} / {{renderValue(gs.player_armor)}}</div>
                                <div className="field"><span className="field-name">Money</span>{{renderValue(gs.player_money)}}</div>
                                <div className="field"><span className="field-name">Team Money</span>{{renderValue(gs.team_money_total)}}</div>
                                <div className="field"><span className="field-name">Primary</span>{{renderValue(gs.weapon_primary)}}</div>
                                <div className="field"><span className="field-name">Secondary</span>{{renderValue(gs.weapon_secondary)}}</div>
                                <div className="field"><span className="field-name">Utility</span>{{renderValue(gs.utility)}}</div>
                                <div className="field"><span className="field-name">Alive</span>{{renderValue(gs.alive_teammates)}} vs {{renderValue(gs.alive_enemies)}}</div>
                                <div className="field"><span className="field-name">Bomb</span>{{renderValue(gs.bomb_status)}}</div>
                                <div className="field"><span className="field-name">Site</span>{{renderValue(gs.site)}}</div>
                                <div className="field"><span className="field-name">Visible Enemies</span>{{renderValue(gs.visible_enemies)}}</div>
                            </div>

                            <div className="json-section">
                                <div className="section-title">Analysis</div>
                                <div className="text-block">{{an.situation_summary || 'N/A'}}</div>
                                <div className="field"><span className="field-name">Economy</span>{{renderValue(an.economy_assessment)}}</div>
                                <div className="field"><span className="field-name">Round Importance</span>{{renderValue(an.round_importance)}}</div>
                                {{an.immediate_threats?.length > 0 && (
                                    <div style={{{{marginTop: '8px'}}}}>
                                        <div className="field-name">Threats</div>
                                        {{an.immediate_threats.map((t, i) => <div key={{i}} className="threat-item">{{t}}</div>)}}
                                    </div>
                                )}}
                                {{an.opportunities?.length > 0 && (
                                    <div style={{{{marginTop: '8px'}}}}>
                                        <div className="field-name">Opportunities</div>
                                        {{an.opportunities.map((o, i) => <div key={{i}} className="opportunity-item">{{o}}</div>)}}
                                    </div>
                                )}}
                            </div>

                            <div className="json-section">
                                <div className="section-title">Advice</div>
                                <div className="text-block" style={{{{background: '#1a3a1a'}}}}>{{ad.primary_action || 'N/A'}}</div>
                                <div style={{{{marginTop: '8px'}}}}>
                                    <div className="field-name">Reasoning</div>
                                    <div className="text-block">{{ad.reasoning || 'N/A'}}</div>
                                </div>
                                <div style={{{{marginTop: '8px'}}}}>
                                    <div className="field-name">Fallback</div>
                                    <div className="text-block">{{ad.fallback || 'N/A'}}</div>
                                </div>
                                <div className="field" style={{{{marginTop: '8px'}}}}><span className="field-name">Callout</span>{{renderValue(ad.callout)}}</div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }}

        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML viewer for reviewing screenshots + labels"
    )
    parser.add_argument(
        "--images", "-i",
        type=str,
        default="data/raw",
        help="Directory containing images (default: data/raw)"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default="data/labeled",
        help="Directory containing Claude JSON labels (default: data/labeled)"
    )
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        default="data/predictions",
        help="Directory containing model predictions (default: data/predictions)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="review.html",
        help="Output HTML file (default: review.html)"
    )
    parser.add_argument(
        "--embed", "-e",
        action="store_true",
        help="Embed images as base64 (larger file, but works without a server)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Generate comparison view (Claude vs Qwen predictions)"
    )

    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    predictions_dir = Path(args.predictions)
    output_path = Path(args.output)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return 1

    if args.compare:
        # Comparison mode: show Claude vs Qwen side by side
        comparisons = find_comparison_data(images_dir, labels_dir, predictions_dir)
        with_predictions = sum(1 for c in comparisons if c["qwen"] is not None)
        print(f"Found {len(comparisons)} labeled images ({with_predictions} with Qwen predictions)")

        if not comparisons:
            print("No labeled images found!")
            return 1

        html = generate_comparison_html(comparisons, embed_images=args.embed)
    else:
        # Standard review mode
        pairs = find_pairs(images_dir, labels_dir)
        print(f"Found {len(pairs)} image-label pairs")

        if not pairs:
            print("No matching pairs found!")
            print(f"  Images in {images_dir}: {len(list(images_dir.glob('*')))}")
            print(f"  Labels in {labels_dir}: {len(list(labels_dir.glob('*.json')))}")
            return 1

        html = generate_html(pairs, embed_images=args.embed)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated {output_path}")

    if args.embed:
        print(f"\nOpen directly in browser:")
        print(f"  open {output_path}")
    else:
        print(f"\nTo view, start a local server:")
        print(f"  python -m http.server 8000")
        print(f"  Then open http://localhost:8000/{output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
