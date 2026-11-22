"""Pick real or synthetic data source from settings. The only place that branches on mode."""

from mono_agent.config import Settings
from mono_agent.mono.client import MonoClient
from mono_agent.mono.ratelimit import Unlimited


def build_client(settings: Settings) -> MonoClient:
    if settings.mono_token is not None:
        return MonoClient(settings.mono_token.get_secret_value(), settings.mono_base_url)

    from mono_agent.synthetic.fake_api import FakeMonoApi
    from mono_agent.synthetic.generator import generate

    data = generate(settings.synthetic_seed)
    api = FakeMonoApi(data.client_info, data.statements, data.rates)
    # No rate limiting against the in-process fake: nothing to protect, and tests stay fast.
    buckets = {k: Unlimited() for k in ("client-info", "statement", "currency")}
    return MonoClient("synthetic", transport=api.transport, buckets=buckets)
