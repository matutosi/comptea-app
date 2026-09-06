# https://discuss.streamlit.io/t/redirecting-logger-output-to-a-streamlit-widget/61070/3
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

@contextmanager
def st_capture_stdout(output_func):
    with io.StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write
        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        stdout.write = new_write
        yield

@contextmanager
def st_capture_stderr(output_func):
    with io.StringIO() as stderr, redirect_stderr(stderr):
        old_write = stderr.write
        def new_write(string):
            ret = old_write(string)
            output_func(stderr.getvalue())
            return ret
        stderr.write = new_write
        yield

if __name__ == "__main__":

    import streamlit as st
    from time import sleep

    # show on browser
    output = st.empty()
    with st_capture_stdout(output.code):
        print("Hello")
        sleep(3)
        print("World")

    output = st.empty()
    with st_capture_stderr(output.info):
        raise ValueError("The first error")
        sleep(3)
        raise ValueError("The second error")

    # show on console
    print("Goodbye")
    raise ValueError("The third error")
