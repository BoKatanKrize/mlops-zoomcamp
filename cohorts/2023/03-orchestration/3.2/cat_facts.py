import httpx
from prefect import flow, task


@task(retries=4,                # Retry the task up to 4 times if the task were to fail for some reason
      retry_delay_seconds=0.1,  # Between each retry, wait a short time
      log_prints=True)          # any print statement will be shared within the logs
def fetch_cat_fact():
    cat_fact = httpx.get("https://f3-vyx5c2hfpq-ue.a.run.app/")
    # An endpoint that is designed to fail sporadically
    # just to demonstrate how the retry logic works
    if cat_fact.status_code >= 400:
        raise Exception()
    print(cat_fact.text)


@flow                            # The flow just calls the task above
def fetch():
    fetch_cat_fact()


if __name__ == "__main__":
    fetch()