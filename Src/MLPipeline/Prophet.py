# Import necessary libraries
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot
import pickle

class Propht:
    def __init__(self, df):
        self.exec(df)

    def exec(self, df):
        # Create a Prophet model instance
        fb_model = Prophet()

        # Fit the model using the provided dataframe
        fb_model.fit(df)

        # Generate future dates for forecasting
        future_dates_healthcare = fb_model.make_future_dataframe(periods=12, freq='M')

        # Predict the range for future dates
        forecast = fb_model.predict(future_dates_healthcare)

        # Display the last few rows of the forecast
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        # Plot the forecast with uncertainty
        fb_model.plot(forecast, uncertainty=True)

        # Plot the components of the forecast
        fb_model.plot_components(forecast)

        # Create a figure to plot the components
        fig1 = fb_model.plot_components(forecast)

        # Add changepoints to the forecast plot
        fig = fb_model.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), fb_model, forecast)

        # Display the changepoints of the model
        print(fb_model.changepoints)

        # Modify the changepoint range
        pro_change = Prophet(changepoint_range=0.5)
        forecast = pro_change.fit(df).predict(future_dates_healthcare)
        fig = pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

        # Set a specific number of changepoints and enable yearly seasonality
        pro_change = Prophet(n_changepoints=20, yearly_seasonality=True)
        forecast = pro_change.fit(df).predict(future_dates_healthcare)
        fig = pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

        # Adjust the trend by modifying changepoint_prior_scale
        pro_change = Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
        forecast = pro_change.fit(df).predict(future_dates_healthcare)
        fig = pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

        # Adjust the trend with a different changepoint_prior_scale
        pro_change = Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.001)
        forecast = pro_change.fit(df).predict(future_dates_healthcare)
        fig = pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

        # Save the trained Prophet model to a file
        with open("../Output/models/fb_model", "wb") as f:
            pickle.dump(fb_model, f)
