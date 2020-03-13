import os
import urllib


ENABLED = os.environ['ENABLED']  # Whether to enable posting of Slack messages.
# Slack OAuth access token from environment variables
ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
CHANNEL = os.environ['CHANNEL']  # Slack channel to post message to


def lambda_handler(event, context):
    message = event['message']
    # message = "hello"
    if ENABLED == 'true':
        post_message(message)
    else:
        return 1
    return event


def post_message(message):
    """ Posts message to Slack channel via Slack API.
    Args:
        message (string): Message to post to channel
    Returns:
        (None)
    """
    url = 'https://slack.com/api/chat.postMessage'
    data = urllib.parse.urlencode(
        (
            ("token", ACCESS_TOKEN),
            ("channel", CHANNEL),
            ("text", message)
        )
    )
    data = data.encode("ascii")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    request = urllib.request.Request(url, data, headers)
    urllib.request.urlopen(request)
