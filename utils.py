import time
import functools
from ttt.config import logger


def retry(func=None, exception=Exception, n_tries=5, delay=0.1, backoff=2, logger=True, on_failure=None):
    """Retry decorator with exponential backoff.
    https://stackoverflow.com/questions/42521549/retry-function-in-python

    Parameters
    ----------
    func : typing.Callable, optional
        Callable on which the decorator is applied, by default None
    exception : Exception or tuple of Exceptions, optional
        Exception(s) that invoke retry, by default Exception
    n_tries : int, optional
        Number of tries before giving up, by default 5
    delay : int, optional
        Initial delay between retries in seconds, by default 0.1
    backoff : int, optional
        Backoff multiplier e.g. value of 2 will double the delay, by default 1
    logger : bool, optional
        Option to log or print, by default True

    Returns
    -------
    typing.Callable
        Decorated callable that calls itself when exception(s) occur.

    Examples
    --------
    ... import random
    ... @retry(exception=Exception, n_tries=4)
    ... def test_random(text):
    ...    x = random.random()
    ...    if x < 0.5:
    ...        raise Exception("Fail")
    ...    else:
    ...        print("Success: ", text)
    ... test_random("It works!")
    """
    # Not sure why this is here
    if func is None:
        return functools.partial(
            retry,
            exception=exception,
            n_tries=n_tries,
            delay=delay,
            backoff=backoff,
            logger=logger,
            on_failure=on_failure,
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ntries, ndelay = n_tries, delay
        exe = None
        while ntries > 0:
            try:
                return func(*args, **kwargs)
            except exception as e:
                exe = e
                msg = f"Failed with exception: {str(e)}, Retrying in {ndelay} seconds..."
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
                time.sleep(ndelay)
                ntries -= 1
                ndelay *= backoff

        if on_failure is not None:
            on_failure(*args, **kwargs)
        else:
            raise exe

    return wrapper


