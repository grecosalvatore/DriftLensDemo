document.addEventListener("DOMContentLoaded", function () {
  let loaderWrapper = document.querySelector(".loader-wrapper");
  let loaderPercentage = document.querySelector(".loader-percentage");
  let loaderText = document.querySelector(".loader-text");
  let hero = document.querySelector(".hero");
  let currentCount = 0; // Initialize the counter

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

  // Show the loader and loading message initially
  if (loaderWrapper && loaderPercentage && loaderText) {
    loaderWrapper.style.display = "block";
    loaderPercentage.style.display = "block";
    loaderText.style.display = "block";
    hero.style.display = "none";
  }

  function createGradient(svg, id, startColor, endColor) {
    let defs = svg.querySelector('defs') || svg.insertBefore(document.createElementNS('http://www.w3.org/2000/svg', 'defs'), svg.firstChild);
    let linearGradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    linearGradient.setAttribute('id', id);
    linearGradient.setAttribute('x1', '0%');
    linearGradient.setAttribute('y1', '0%');
    linearGradient.setAttribute('x2', '100%');
    linearGradient.setAttribute('y2', '100%');
    let stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', startColor);
    linearGradient.appendChild(stop1);
    let stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('stop-color', endColor);
    linearGradient.appendChild(stop2);
    defs.appendChild(linearGradient);
  }


  function updateLoaderPercentage(percentage) {
    const radius = 35; // Radius of the circle
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percentage / 100) * circumference;

    const loaderCircle = document.querySelector('.loader-circle');
    if (loaderCircle) {
      let svg = loaderCircle.nearestViewportElement;
      createGradient(svg, 'loader-gradient', "rgba(78, 42, 132, 0.7)", "rgba(215, 38, 56, 0.7)"); // Replace with your colors
      loaderCircle.style.stroke = 'url(#loader-gradient)';
      loaderCircle.style.strokeDashoffset = offset;
    }

    const loaderPercentage = document.querySelector('.loader-percentage');
    if (loaderPercentage) {
      loaderPercentage.innerText = `${percentage.toFixed(0)}%`;
    }
  }


  // Initially set loader percentage to 0%
  updateLoaderPercentage(0);

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

  socket.on("UpdateProgressBarDataStreamGeneration", function (jsonString) {
    // Parse the JSON string
    const data = JSON.parse(jsonString);

    console.log("Received progressData :: " + data.currentProgress);
    if (data && data.currentProgress !== undefined) {
      updateLoaderPercentage(data.currentProgress);
    }
  });

  // Receive details from the server
  socket.on("updateSensorData", function (msg) {
    console.log("Received sensorData :: " + msg.date + " :: " + msg.batch_distance + " :: " + msg.per_label_distances);

    // Hide the loader and loading message when data is received
    if (loaderWrapper && loaderPercentage && loaderText) { // Add loaderText here
      loaderWrapper.style.display = "none";
      loaderPercentage.style.display = "none";
      loaderText.style.display = "none"; // Hide the text
      hero.style.display = "block";
    }

    // Show only MAX_DATA_COUNT data
    if (myChart.data.labels.length > MAX_DATA_COUNT) {
      removeFirstData();
    }
    addData(msg.date, msg.batch_distance, msg.per_label_distances);
  });
});
