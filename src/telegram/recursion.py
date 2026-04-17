import os, asyncio, json, logging
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from langdetect import detect, DetectorFactory
import networkx as nx

from dotenv import load_dotenv
load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE")
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 2))
SEED_CHANNELS = os.getenv("SEED_CHANNELS").strip("[]").replace("'", "").split(", ")
BOT_LOG = "crawler.log"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(BOT_LOG), logging.StreamHandler()])
logger = logging.getLogger()
DetectorFactory.seed = 0

client = TelegramClient('session', API_ID, API_HASH)

def is_english(text, min_len=20):
    if not text or len(text) < min_len:
        return False
    try:
        return detect(text) == "en"
    except:
        return False

async def crawl_depth(channels, depth, graph):
    if depth > MAX_DEPTH or not channels:
        return set()

    await client.start(PHONE)
    new_channels = set()

    for chan in channels:
        logger.info(f"[Depth {depth}] Processing @{chan}")
        try:
            await client(JoinChannelRequest(chan))
        except Exception as e:
            logger.warning(f"Can't join {chan}: {e}")
            continue

        async for msg in client.iter_messages(chan, limit=200):
            if not is_english(msg.text or ""):
                continue

            if msg.forward and msg.forward.chat and msg.forward.chat.username:
                src = msg.forward.chat.username
                graph.add_node(chan)
                graph.add_node(src)
                graph.add_edge(chan, src)
                if src not in graph:
                    new_channels.add(src)

    await client.disconnect()
    logger.info(f"[Depth {depth}] Discovered {len(new_channels)} new channels")
    return new_channels

async def main():
    G = nx.DiGraph()
    current = set(SEED_CHANNELS)
    all_seen = set(current)

    for depth in range(0, MAX_DEPTH + 1):
        logger.info(f"=== Starting depth {depth} crawl with {len(current)} channels")
        discovered = await crawl_depth(current, depth, G)
        discovered -= all_seen
        all_seen |= discovered
        current = discovered

    nx.write_gexf(G, "tg_channel_network.gexf")
    json.dump(list(G.nodes), open("nodes.json","w"), indent=2)
    json.dump(list(G.edges), open("edges.json","w"), indent=2)
    logger.info(f"Completed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

with client:
    client.loop.run_until_complete(main())