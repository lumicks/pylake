def check_widget_backend():
    import matplotlib.pyplot as plt

    if not max(
        # Note: Some, but not all versions of matplotlib lower the backend names. Hence, we
        # always lower them to be on the safe side.
        [backend in plt.get_backend().lower() for backend in ("nbagg", "ipympl", "widget")]
    ):
        raise RuntimeError(
            (
                "Please enable an interactive matplotlib backend for this widget to work. In "
                "jupyter notebook or lab you can do this by invoking either "
                "%matplotlib widget or %matplotlib ipympl. Please note that you may have to "
                "restart the notebook kernel for this to work."
            ),
        )
