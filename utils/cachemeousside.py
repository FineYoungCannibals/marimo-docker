import asyncio
import httpx
import marimo as mo
import nest_asyncio
import os
import re
import time
from urllib.parse import urlparse
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

nest_asyncio.apply()

cache_base_url = os.getenv("CACHEMEOUSSIDE_API_URL")
allowed_sources = {
    "ipqs": "ipqs/domain_lookup",
    "urlscan": "urlscan/domain_lookup"
}


async def check(query, source, days=30):
    """Make an asynchronous HTTP request to cache URL API."""
    if source not in allowed_sources.keys():
        raise ValueError(f"Invalid source {source}, must be one of {allowed_sources}")

    params = {"query": query, "days": days}

    async with httpx.AsyncClient(verify=False) as client:
        cache_url = f"{cache_base_url}/{allowed_sources[source]}"
        response = await client.get(cache_url, params=params)
        return response.json()


async def check_multiple(queries, source, retry_delay=5, days=30):
    """Run multiple HTTP requests concurrently with Marimo progress bar."""
    if source not in allowed_sources.keys():
        raise ValueError(f"Invalid source {source}, must be one of {allowed_sources}")

    start_time = time.time()
    print(f"Starting requests for {len(queries)} queries")

    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        cache_url = f"{cache_base_url}/{allowed_sources[source]}"

        # First pass - send all requests concurrently
        first_pass_tasks = []
        for query in queries:
            params = {"query": query, "days": days}
            task = asyncio.create_task(client.get(cache_url, params=params))
            first_pass_tasks.append((query, task))

        # Wait for all first-pass tasks to complete or fail
        results = [None] * len(queries)
        failed_indices = []

        # Use Marimo's progress_bar for tracking first pass
        for i, (query, task) in enumerate(mo.status.progress_bar(
            first_pass_tasks,
            title="Processing Queries",
            subtitle="First pass",
            show_eta=True
        )):
            try:
                response = await task
                results[i] = response.json()
            except Exception as e:
                print(f"Initial error for {query}: {type(e).__name__}")
                failed_indices.append(i)

        # If any failed, wait then retry those concurrently
        if failed_indices:
            print(f"Waiting {retry_delay}s before retrying {len(failed_indices)} queries...")
            await asyncio.sleep(retry_delay)

            retry_tasks = []
            for i in failed_indices:
                query = queries[i]
                params = {"query": query}
                task = asyncio.create_task(client.get(cache_url, params=params))
                retry_tasks.append((i, query, task))

            # Process retry results with another progress bar
            if retry_tasks:
                for idx, (i, query, task) in enumerate(mo.status.progress_bar(
                    retry_tasks,
                    title="Retrying Failed Queries",
                    subtitle=f"Retrying {len(retry_tasks)} queries",
                    show_eta=True
                )):
                    try:
                        response = await task
                        results[i] = response.json()
                        print(f"Retry success for {query}")
                    except Exception as e:
                        print(f"Retry failed for {query}: {type(e).__name__}")
                        results[i] = {"error": str(e), "query": query}

    total_time = time.time() - start_time
    print(f"Completed all requests in {total_time:.2f}s")
    return results


def convert_url_to_domain(url):
    """Given a URL, return on ly the domain portion"""
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc or parsed_url.path
    netloc = netloc.split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    domain_pattern = r'([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}'
    match = re.search(domain_pattern, netloc)
    if match:
        return match.group()
    else:
        return None