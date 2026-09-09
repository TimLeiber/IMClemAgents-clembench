"""Create vanilla Clembench HTML transcripts with provider-exposed reasoning."""

import argparse
import glob
import html
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import markdown as md

from clemcore.clemgame.resources import load_json, store_file
from clemcore.clemgame.transcripts import html_templates
from clemcore.clemgame.transcripts.builder import _get_class_name, get_css, get_css_player_dict
from clemcore.utils import file_utils


REASONING_CSS = """
.msg.reasoning {
  background: #fff7d6;
  border-left: .35rem solid #c58d00;
  border-radius: var(--rad-sm);
  color: #3b2e00;
  font-family: monospace;
  font-size: .9rem;
  margin-left: 12%;
  margin-right: 12%;
}
"""


def text_from_value(value):
    """Return a non-empty text value when one is available."""

    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list):
        text_parts = [text_from_value(item) for item in value]
        text_parts = [text for text in text_parts if text]
        if text_parts:
            return "\n".join(text_parts)
    if isinstance(value, dict):
        for key in ["text", "content", "summary"]:
            text = text_from_value(value.get(key))
            if text:
                return text
    return None


def extract_reasoning(raw_response):
    """Extract provider-exposed reasoning from an OpenAI-style response object."""

    if not isinstance(raw_response, dict):
        return None
    choices = raw_response.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return None
    for key in ["reasoning", "reasoning_content", "thinking", "analysis"]:
        reasoning = text_from_value(message.get(key))
        if reasoning:
            return reasoning
    return text_from_value(message.get("reasoning_details"))


def load_reasoning_by_player_round(instance_dir):
    """Load reasoning and align it with the player response made in the same round."""

    reasoning_by_player_round = defaultdict(deque)
    for request_path in instance_dir.glob("player_*.requests.json"):
        request_log = load_json(str(request_path))
        player_name = request_log.get("meta", {}).get("player_name")
        if not player_name:
            continue
        for request in request_log.get("calls", []):
            raw_response = request.get("call", {}).get("raw_response_obj")
            reasoning = extract_reasoning(raw_response)
            if reasoning:
                key = (request.get("round"), player_name)
                reasoning_by_player_round[key].append(reasoning)
    return reasoning_by_player_round


def render_event(interactions,
                 event,
                 css_player_dict,
                 markdown):
    """Render one regular Clembench interaction event using its normal layout."""

    players = interactions["players"]
    class_name, player = _get_class_name(event, css_player_dict)
    if player is not None:
        class_name += f" {player}"
    msg_content = event["action"]["content"]
    if markdown:
        msg_raw = msg_content.strip()
        while msg_raw.startswith("`") and msg_raw.endswith("`"):
            msg_raw = msg_raw[1:-1]
        msg_raw = md.markdown(msg_raw, extensions=["fenced_code"])
    else:
        msg_raw = html.escape(f"{msg_content}").replace("\n", "<br/>")
    if event["from"] == "GM" and event["to"] == "GM":
        speaker_attr = f'Game Master: {event["action"]["type"]}'
    else:
        from_player = event["from"]
        to_player = event["to"]
        if "game_role" in players[from_player] and "game_role" in players[to_player]:
            speaker_attr = (f'{from_player} ({players[from_player]["game_role"]}) '
                            f'to {to_player} ({players[to_player]["game_role"]})')
        else:
            speaker_attr = f'{from_player.replace("GM", "Game Master")} to {to_player.replace("GM", "Game Master")}'
    if isinstance(msg_content, str):
        try:
            msg_content = json.loads(msg_content)
        except json.JSONDecodeError:
            pass
    style = "border: dashed" if event["action"].get("label") == "forget" else ""
    images = list(event["action"].get("image", []))
    if isinstance(msg_content, dict) and "image" in msg_content:
        images += msg_content["image"]
    if not images:
        return html_templates.EVENT_TEMPLATE.format(speaker_attr, class_name, style, msg_raw)
    event_html = f'<div speaker="{speaker_attr}" class="msg {class_name}" style="{style}">\n'
    event_html += f"  <p>{msg_raw}</p>\n"
    for image_src in images:
        if not image_src.startswith("http"):
            if "IMAGE_ROOT" in os.environ:
                image_src = os.path.join(os.environ["IMAGE_ROOT"], image_src)
            elif not image_src.startswith("/"):
                image_src = os.path.join(file_utils.project_root(), image_src)
        event_html += f'  <a title="{image_src}"><img style="width:100%" src="{image_src}" alt="{image_src}" /></a>\n'
    return event_html + "</div>\n"


def build_transcript_with_reasoning(interactions,
                                    reasoning_by_player_round):
    """Create the regular HTML transcript with reasoning before each matching response."""

    meta = interactions["meta"]
    players = interactions["players"]
    markdown = interactions.get("markdown", False)
    css_player_dict = get_css_player_dict(players)
    transcript = html_templates.HEADER.format(get_css(len(players)) + REASONING_CSS)
    pair_descriptor = meta.get("results_folder", meta.get("dialogue_pair", ""))
    title = (f"Interaction Transcript with provider-exposed reasoning for game '{meta['game_name']}', "
             f"experiment '{meta['experiment_name']}', episode {meta['game_id']} with {pair_descriptor}.")
    transcript += html_templates.TOP_INFO.format(title)
    for turn_idx, turn in enumerate(interactions["turns"]):
        transcript += f'<div class="game-round" data-round="{turn_idx}">'
        for event in turn:
            reasoning_entries = reasoning_by_player_round.get((turn_idx, event["from"]))
            if reasoning_entries:
                reasoning = reasoning_entries.popleft()
                reasoning_html = html.escape(reasoning).replace("\n", "<br/>")
                speaker = f'{event["from"]} reasoning (provider-exposed)'
                transcript += (f'<div speaker="{speaker}" class="msg reasoning">'
                               f"<p>{reasoning_html}</p></div>\n")
            transcript += render_event(interactions, event, css_player_dict, markdown)
        transcript += "</div>"
    return transcript + html_templates.FOOTER


def transcribe_reasoning(results_dir,
                         game):
    """Write transcript_reasoning.html next to every matching interactions.json file."""

    interaction_files = glob.glob(str(Path(results_dir) / "**" / "interactions.json"), recursive=True)
    if game != "all":
        interaction_files = [path for path in interaction_files if game in Path(path).parts]
    for interaction_path in interaction_files:
        instance_dir = Path(interaction_path).parent
        interactions = load_json(interaction_path)
        reasoning_by_player_round = load_reasoning_by_player_round(instance_dir)
        transcript = build_transcript_with_reasoning(interactions, reasoning_by_player_round)
        store_file(transcript, "transcript_reasoning.html", instance_dir)
    print(f"Generated {len(interaction_files)} reasoning transcripts")


def main():
    """Parse command-line arguments and create reasoning transcripts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results_dir", default="results")
    parser.add_argument("-g", "--game", default="all")
    args = parser.parse_args()
    transcribe_reasoning(args.results_dir, args.game)


if __name__ == "__main__":
    main()
