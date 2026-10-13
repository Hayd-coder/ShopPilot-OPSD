#!/usr/bin/env python3
"""CPU-only regression test for WebShop selected-state trace metadata."""

from __future__ import annotations

import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
webshop_root = (
    repo_root / "agent_system/environments/env_package/webshop/webshop"
)
sys.path.insert(0, str(webshop_root))
sys.path.insert(0, str(repo_root))

from web_agent_site.envs import WebAgentTextEnv  # noqa: E402
from agent_system.environments.env_manager import WebshopEnvironmentManager  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def get_title(product: dict) -> str:
    return str(product.get("Title") or product.get("title") or product.get("name") or "")


def get_options(product: dict) -> dict:
    return product.get("options") or product.get("Options") or {}


def main() -> None:
    env = WebAgentTextEnv(
        observation_mode="text",
        num_products=1000,
        human_goals=False,
        seed=2026,
        file_path=str(webshop_root / "data/items_shuffle_1000.json"),
        attr_path=str(webshop_root / "data/items_ins_v2_1000.json"),
        session_prefix="state-trace-test-",
    )
    env.reset(session=0)

    candidates = [
        product
        for product in env.server.all_products
        if get_title(product).strip() and get_options(product)
    ]
    assert candidates, "No 1k catalogue product with title and options was found"

    chosen = None
    for product in candidates:
        title = get_title(product)
        env.step(f"search[{title}]")
        actions = env.get_available_actions()["clickables"]
        asin = str(product["asin"]).lower()
        if asin in actions:
            chosen = product
            break
    assert chosen is not None, "No option-bearing product was retrievable by exact title"

    asin = str(chosen["asin"]).lower()
    _, _, done, product_info = env.step(f"click[{asin}]")
    assert not done
    assert str(product_info["selected_asin"]).lower() == asin
    assert product_info["selected_options"] == {}

    clickables = env.get_available_actions()["clickables"]
    option_values = [
        str(value).lower()
        for values in get_options(chosen).values()
        for value in (values if isinstance(values, list) else [values])
    ]
    option_action = next(
        (value for value in option_values if value in clickables),
        None,
    )
    assert option_action is not None, (
        f"No product option was clickable; options={option_values[:5]}, "
        f"clickables={clickables[:15]}"
    )

    _, reward, done, option_info = env.step(f"click[{option_action}]")
    assert not done
    assert reward == 0
    assert option_info["selected_options"], option_info
    assert option_info["reward_breakdown"] == {}

    # Select one value for every remaining option group, then purchase.  The
    # terminal metadata must survive WebAgentTextEnv's automatic reset.
    for option_name, values in get_options(chosen).items():
        if option_name.lower() in option_info["selected_options"]:
            continue
        available = env.get_available_actions()["clickables"]
        values = values if isinstance(values, list) else [values]
        value = next(
            (str(candidate).lower() for candidate in values if str(candidate).lower() in available),
            None,
        )
        if value is not None:
            _, _, _, option_info = env.step(f"click[{value}]")

    assert "buy now" in env.get_available_actions()["clickables"]
    _, terminal_reward, terminal_done, terminal_info = env.step("click[buy now]")
    assert terminal_done
    assert 0.0 <= terminal_reward <= 1.0
    assert terminal_info["selected_asin"]
    assert terminal_info["selected_options"]
    assert terminal_info["reward_breakdown"], terminal_info

    manager = WebshopEnvironmentManager.__new__(WebshopEnvironmentManager)
    manager.tasks = ["buy a black shirt in size x-small"]
    manager.config = OmegaConf.create(
        {
            "env": {
                "history_length": 2,
                "webshop": {"expose_selected_options": True},
            }
        }
    )
    prompt = manager.build_text_obs(
        ["Item page with size and color choices"],
        [
            {
                "available_actions": {
                    "has_search_bar": True,
                    "clickables": ["x-small", "black", "buy now"],
                },
                "selected_asin": terminal_info["selected_asin"],
                "selected_options": terminal_info["selected_options"],
            }
        ],
        init=True,
    )[0]
    assert "CURRENT BROWSER STATE" in prompt
    assert "size=x-small" in prompt
    assert "color=black" in prompt
    assert "only entries in selected_options have actually been selected" in prompt

    print(f"asin={asin}")
    print(f"selected_options={terminal_info['selected_options']}")
    print(f"reward_breakdown={terminal_info['reward_breakdown']}")
    print("WebShop selected-state trace metadata: PASS")
    print("WebShop selected-state prompt: PASS")


if __name__ == "__main__":
    main()
