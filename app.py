from flask import Flask, render_template
import plotly.express as px

app = Flask(__name__)

@app.route("/")
def home():
    title = 'DriftLens'
    return render_template('home.html', title=title)

@app.route("/drift_lens_monitor")
def plotly_chart():
    # Create or retrieve data for your Plotly chart
    data = [1, 2, 3, 4, 5]

    # Create the Plotly chart
    fig = px.line(x=list(range(1, len(data) + 1)), y=data, title='Your Plotly Chart')

    # Generate HTML for the Plotly chart
    plotly_chart_div = fig.to_html(full_html=False)

    return render_template('drift_lens_monitor.html', plotly_chart=plotly_chart_div)

if __name__ == '__main__':
    app.run(debug=True)
