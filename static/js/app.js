$(document).ready(function () {
  const ctx = document.getElementById("myChart").getContext("2d");
  const ctx_per_label = document.getElementById("myChart_per_label").getContext("2d");

  const myChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [], // Initialize labels array for per-batch chart
      datasets: [{ label: "per-batch distance", data: [], }],
    },
    options: {
      scales: {
        y: {
          beginAtZero: true, // You can customize other y-axis options here
          title: {
            display: true,
            text: "Distribution Distance", // Set the label text here
          },
        },
      },
      borderWidth: 3,
      borderColor: ["rgba(255, 99, 132, 1)"],
    },
  });

  const numLabels = parseInt($("#num_labels").data("num-labels")); // Get the number of labels from the HTML
  const labelNames = $("#label_names").data("label-names").split(","); // Get label names from the HTML and split into an array

  const myChart_per_label = new Chart(ctx_per_label, {
    type: "line",
    data: {
      labels: [], // Initialize labels array for per-label chart
      datasets: Array.from({ length: numLabels }, (_, i) => ({
        label: labelNames[i], // Assign label names from the labelNames array
        data: [], // Initialize with empty data
        borderColor: getRandomColor(),
        backgroundColor: getRandomColor(),
        fill: false,
      })),
    },
    options: {
      scales: {
        y: {
          beginAtZero: true, // You can customize other y-axis options here
          title: {
            display: true,
            text: "Distribution Distance", // Set the label text here
          },
        },
      },
      borderWidth: 3,
    },
  });

  function getRandomColor() {
    const randomColor = "#" + Math.floor(Math.random() * 16777215).toString(16);
    return randomColor;
  }

  function addData(label, batch_distance, per_label_distances) {
    myChart.data.labels.push(label);
    myChart.data.datasets.forEach((dataset) => {
      dataset.data.push(batch_distance);
    });

    // Split per-label distances into an array
    const perLabelValues = per_label_distances.split(",").map(Number);

    // Update per-label chart
    myChart_per_label.data.labels.push(label);
    myChart_per_label.data.datasets.forEach((dataset, i) => {
      dataset.data.push(perLabelValues[i]);
    });

    // Update both charts
    myChart.update();
    myChart_per_label.update();
  }

  function removeFirstData() {
    // Remove the first data point from per-batch chart
    myChart.data.labels.splice(0, 1);
    myChart.data.datasets[0].data.shift();

    // Remove the first data point from per-label chart
    myChart_per_label.data.labels.splice(0, 1);
    myChart_per_label.data.datasets.forEach((dataset) => {
      dataset.data.shift();
    });

    // Update both charts
    myChart.update();
    myChart_per_label.update();
  }

  const MAX_DATA_COUNT = 1000;
  // Connect to the socket server
  var socket = io.connect();

  // Receive details from the server
  socket.on("updateSensorData", function (msg) {
    console.log("Received sensorData :: " + msg.date + " :: " + msg.batch_distance + " :: " + msg.per_label_distances);

    // Show only MAX_DATA_COUNT data
    if (myChart.data.labels.length > MAX_DATA_COUNT) {
      removeFirstData();
    }
    addData(msg.date, msg.batch_distance, msg.per_label_distances);
  });
});
