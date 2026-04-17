import os, json, logging
import re
import emoji
import socks
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from langdetect import detect, DetectorFactory
import networkx as nx
from telethon.errors import UserNotParticipantError, InviteRequestSentError, ChannelPrivateError, ChannelInvalidError
from telethon.tl.functions.channels import GetParticipantRequest
from bertopic import BERTopic
import asyncio
from telethon.errors import FloodWaitError, RpcCallFailError
from telethon.tl.types import Message


# ——— Environment & Configuration —————————————————————————
load_dotenv()
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE")

decodo_proxy = (
    socks.SOCKS5,               # or socks.HTTP
    'gate.decodo.com',         # Decodo proxy host
    7000,                       # port
    True,                       # rdns
    os.getenv("DECODO_USER"),         # username
    os.getenv("DECODO_PASSWORD")          # password
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

DetectorFactory.seed = 0    # deterministic lang detect

seed_channels_raw = os.getenv("SEED_CHANNELS_1")
SEED_CHANNELS = seed_channels_raw.strip("[]").replace("'", "").split(", ")

# Output files
API_JSON = "api_messages.json"
SCRAPE_JSON = "scraped_messages.json"

client = TelegramClient('session', API_ID, API_HASH, proxy=decodo_proxy)

# ——— Language Filter ———————————————————————————————
def is_english(text):
    if not text:
        return False
    try:
        return detect(text) == "en"
    except:
        return False

# ——— Join Channel Safely ———————————————————————————————
async def safe_join(chan, invite_only_list):
    try:
        await client(GetParticipantRequest(channel=chan, participant='me'))
        logger.info(f"Already in @{chan}")
    except UserNotParticipantError:
        try:
            logger.info(f"Joining @{chan}...")
            await client(JoinChannelRequest(chan))
        except InviteRequestSentError:
            logger.warning(f"Invite request sent (awaiting approval): @{chan}")
            invite_only_list.append(chan)
        except ChannelPrivateError:
            logger.error(f"Cannot join @{chan}: channel is private or inaccessible")
        except ChannelInvalidError:
            logger.error(f"Cannot join @{chan}: invalid or non-existent")
        except Exception as e:
            logger.error(f"Unexpected error while joining @{chan}: {type(e).__name__} – {e}")
    except Exception as e:
        logger.error(f"Error verifying membership for @{chan}: {type(e).__name__} – {e}")

# ——— Safe Entity Fetching ———————————————————————————————
async def safe_entity(username):
    try:
        return await client.get_input_entity(username)
    except ValueError as e:
        if f'No user has "{username}" as username' in str(e):
            logger.warning(f"Username @{username} not found, skipping.")
            return None
        raise

# ——— JSON Append ———————————————————————————————
def append_to_json_file(new_records, filename="api_messages.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    if isinstance(data, list):
        data.extend(new_records)
    elif isinstance(data, dict):
        data.update(new_records)
    else:
        raise ValueError("Unexpected JSON structure")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ——— Fetch Comments for Post ———————————————————————————————
async def fetch_comments_for_post(client, channel, msg):
    comments = []

    if msg.replies and msg.replies.replies > 0:
        async for reply in client.iter_messages(channel, reply_to=msg.id):
            comments.append({
                "comment_id": reply.id,
                "date": reply.date.isoformat(),
                "text": reply.text,
                "from_id": getattr(reply.from_id, 'user_id', None),
            })
    return comments

def make_vectorizer(n):
    min_df = max(1, int(0.01 * n))       # at least 2% of docs
    max_df = max(min_df + 1, int(0.95 * n))  # at most 80% of docs
    return CountVectorizer(
        stop_words="english",
        ngram_range=(1,2)
    )

# ——— Check Channel Relevance ———————————————————————————————
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def analyze_topics(messages, min_docs=100, top_n=5):
    docs = [m.text for m in messages if m.text and is_english(m.text)]
    def clean_text(text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r"u/\w+|r/\w+|>\s.*", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = text.lower()
        return text
    docs = [clean_text(doc) for doc in docs]

    if int(len(docs)) < min_docs:
        logger.info(f"Only {len(docs)} docs—skipping topic analysis.")
        return None  # not enough data

    #embs = embedding_model.encode(docs, show_progress_bar=False)
    print(f"Analyzing {len(docs)} documents for topics...")
    n = int(len(docs))
    vectorizer_model = make_vectorizer(n)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    repr_model = KeyBERTInspired() # KeyBERTInspired is used to generate more interpretable topic representations

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=repr_model,
        embedding_model="all-mpnet-base-v2",
    )
    #topics, probs = topic_model.fit_transform(docs, embs)

    topics, probs = topic_model.fit_transform(docs)

    info = topic_model.get_topic_info()
    return info.head(top_n)["Representation"].tolist()

def is_relevant(topics, keywords):
    # Flatten topics if it is a list of lists
    if topics and isinstance(topics[0], list):
        topics = [item for sublist in topics for item in sublist]
    text = " ".join(topics).lower()
    return any(k.lower() in text for k in keywords), topics

#model = api.load('glove-wiki-gigaword-100')

#def expand_keywords(base_words, topn=5):
    expanded = set(base_words)
    for w in base_words:
        if w in model:
            for sim_word, _ in model.most_similar(positive=[w], topn=topn):
                expanded.add(sim_word)
    return list(expanded)

keywords = ["longevity", "healthy aging", "healthy ageing", "rejuvenation", "anti-aging", "anti-ageing"]
#keywords = expand_keywords(keywords, topn=10)
keywords_lower = [k.lower() for k in keywords]
print(keywords)
#append_to_json_file(keywords, "keywords.json")

try:
    with open("../take_3/scraped.json", "r") as f:
        scraped_channels = set(json.load(f))
except (FileNotFoundError, json.JSONDecodeError):
    scraped_channels = set()

# ——— Phase 1 Collecting data from seed channels and discovering new groups/channels ————————————————
# Phase 1: discover only
async def phase1_discover(channels, limit_per_channel=100, graph=None, should_crawl=True, output_path="", discovered_output_path=""):
    discovered = set()
    out = []
    invite_only = []
    discarded = []

    await client.start(PHONE)
    for chan in channels:
        if chan in scraped_channels:
            logger.info(f"Skipping @{chan}, already scraped")
            continue

        logger.info(f"[Collection & Discovery] Scanning @{chan}")
        entity = await safe_entity(chan)
        if entity is None:
            continue

        await safe_join(chan, invite_only)

        sample_msgs = await client.iter_messages(chan, limit=500).collect()
        topics = analyze_topics(sample_msgs)
        relevant, labels = is_relevant(topics or [], keywords_lower)
        if not relevant:
            logger.info(f"Skipping @{chan}, topics not relevant: {labels}")
            discarded.append(chan)
            if chan in graph:
                graph.remove_node(chan)
            continue
        logger.info(f"Proceeding @{chan}, topics: {labels}")
        try:
            async for msg in client.iter_messages(chan, limit=limit_per_channel):
                if not is_english(msg.text or ""):
                    continue
                record = {
                    "source": chan,
                    "msg_id": msg.id,
                    "date": msg.date.isoformat(),
                    "text": msg.text,
                    "forwarded_from": None,
                    "comments": []
                }
                if msg.replies and msg.replies.replies > 0:
                    record["comments"] = await fetch_comments_for_post(client, chan, msg)
                if should_crawl:
                    if msg.forward and msg.forward.chat and msg.forward.chat.username:
                        src = msg.forward.chat.username
                        record["forwarded_from"] = src

                        # Add to discovered channels and graph
                        graph.add_node(chan)
                        graph.add_node(src)
                        if graph.has_edge(chan, src):
                            graph[chan][src]['weight'] += 1
                        else:
                            graph.add_edge(chan, src, weight=1)

                        if src not in discovered:
                            logger.info(f"[Collection & Discovery] Discovered forwarded channel: @{src}")
                            discovered.add(src)
                out.append(record)
        except ChannelPrivateError:
            logger.error(f"Access denied to @{chan}, skipping messages")
        except Exception as e:
            logger.error(f"Error fetching messages from @{chan}: {type(e).__name__} – {e}")
        if chan not in scraped_channels:
            scraped_channels.add(chan)
    await client.disconnect()

    append_to_json_file(out, f"{output_path}/messages.json")
    append_to_json_file(list(discovered), f"{output_path}/{discovered_output_path}.json")
    append_to_json_file(invite_only, f"{output_path}/invite_only_channels.json")
    with open(f"{output_path}/scraped.json", "w") as f:
        json.dump(list(scraped_channels), f, indent=2)

    logger.info(f"[Collection & Discovery] Found {len(discovered)} unique forwarded channels")
    logger.info(f"[Collection & Discovery] Discarded {len(discarded)} channels due to irrelevance:" + "\n".join(discarded))
    return discovered

# ——— Main ——————————————————————————————
async def main():
    try:
        G = nx.read_gexf("../take_3/tg_channel_network.gexf")
        logger.info(f"Loaded existing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except:
        G = nx.DiGraph()
        logger.info("Created new graph")

    seed = SEED_CHANNELS
    invite_only = []

    discovered = await phase1_discover(seed, limit_per_channel=None, graph=G, should_crawl=True, output_path="../take_3", discovered_output_path="discovered_channels_1")
    logger.info("Discovered channels:\n" + "\n".join(discovered))

    # with open("../take_3/discovered_channels_2.json", "r", encoding="utf-8") as f:
    #     discovered = set(json.load(f))
    #     logger.info(f"Loaded {len(discovered)} channels from discovered_channels_2.json")
    #
    # discovered3 = await phase1_discover(discovered, limit_per_channel=None, graph=G, should_crawl=True, output_path="../take_3", discovered_output_path="discovered_channels_3")
    # logger.info("Discovered channels:\n" + "\n".join(discovered3))

    with open("../take_3/messages.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    message_count = len(data)
    with open("../take_3/discovered_channels_3.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    forw_channels_count = len(data)
    print(message_count, forw_channels_count)

    nx.write_gexf(G, "../take_3/tg_channel_network.gexf")
    json.dump(list(G.nodes), open("../take_3/nodes.json", "w"), indent=2)
    json.dump(list(G.edges), open("../take_3/edges.json", "w"), indent=2)
    logger.info(f"Completed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

with client:
    client.loop.run_until_complete(main())