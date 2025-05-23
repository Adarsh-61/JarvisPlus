#!/usr/bin python3

import sys
import argparse
import configparser
import asyncio
import os 

from sources.llm_provider import Provider
from sources.interaction import Interaction
from sources.agents import Agent, CoderAgent, CasualAgent, FileAgent, PlannerAgent, BrowserAgent
from sources.browser import Browser, create_driver
from sources.utility import pretty_print

import warnings
warnings.filterwarnings("ignore")

config = configparser.ConfigParser()
config.read('config.ini')

def display_welcome_banner():
    try:
        terminal_width = os.get_terminal_size().columns
    except:
        terminal_width = 80
    
    title = "WELCOME TO JARVIS+"
    subtitle = "NEW MASK, SAME TASK"
    
    box_width = min(terminal_width - 4, max(len(title), len(subtitle)) + 8)
    
    horizontal_line = "+" + "-" * (box_width - 2) + "+"
    empty_line = "|" + " " * (box_width - 2) + "|"
    title_line = "|" + title.center(box_width - 2) + "|"
    subtitle_line = "|" + subtitle.center(box_width - 2) + "|"
    
    banner = [
        horizontal_line,
        empty_line,
        title_line,
        subtitle_line,
        empty_line,
        horizontal_line
    ]
    
    print("\n")
    for line in banner:
        print(line.center(terminal_width))
    print("\n")

async def main():
    pretty_print("Initializing...", color="status")
    display_welcome_banner()  
    stealth_mode = config.getboolean('BROWSER', 'stealth_mode')
    personality_folder = "jarvis" if config.getboolean('MAIN', 'jarvis_personality') else "base"
    languages = config["MAIN"]["languages"].split(' ')

    provider = Provider(provider_name=config["MAIN"]["provider_name"],
                        model=config["MAIN"]["provider_model"],
                        server_address=config["MAIN"]["provider_server_address"],
                        is_local=config.getboolean('MAIN', 'is_local'))

    browser = Browser(
        create_driver(headless=config.getboolean('BROWSER', 'headless_browser'), stealth_mode=stealth_mode),
        anticaptcha_manual_install=stealth_mode
    )

    agents = [
        CasualAgent(name=config["MAIN"]["agent_name"],
                    prompt_path=f"prompts/{personality_folder}/casual_agent.txt",
                    provider=provider, verbose=False),
        CoderAgent(name="coder",
                   prompt_path=f"prompts/{personality_folder}/coder_agent.txt",
                   provider=provider, verbose=False),
        FileAgent(name="File Agent",
                  prompt_path=f"prompts/{personality_folder}/file_agent.txt",
                  provider=provider, verbose=False),
        BrowserAgent(name="Browser",
                     prompt_path=f"prompts/{personality_folder}/browser_agent.txt",
                     provider=provider, verbose=False, browser=browser),
        PlannerAgent(name="Planner",
                     prompt_path=f"prompts/{personality_folder}/planner_agent.txt",
                     provider=provider, verbose=False, browser=browser)
    ]

    interaction = Interaction(agents,
                              tts_enabled=config.getboolean('MAIN', 'speak'),
                              stt_enabled=config.getboolean('MAIN', 'listen'),
                              recover_last_session=config.getboolean('MAIN', 'recover_last_session'),
                              langs=languages
                            )
    try:
        while interaction.is_active:
            interaction.get_user()
            if await interaction.think():
                interaction.show_answer()
    except Exception as e:
        if config.getboolean('MAIN', 'save_session'):
            interaction.save_session()
        raise e
    finally:
        if config.getboolean('MAIN', 'save_session'):
            interaction.save_session()

if __name__ == "__main__":
    asyncio.run(main())