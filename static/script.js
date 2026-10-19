
const fieldsContainer = document.getElementById("feature-fields");
const form = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-button");
const errorMessage = document.getElementById("error-message");

async function loadFeatures() {
    try {
        const response = await fetch("/features");
        if (!response.ok) throw new Error("Could not load model features.");

        const data = await response.json();

        const features = Array.isArray(data) ? data : data.features;

        if (!Array.isArray(features) || features.length === 0) {
            throw new Error("No model features were returned by the API.");
        }

        fieldsContainer.replaceChildren();

        features.forEach((feature) => {
            const field = document.createElement("div");
            field.className = "field";

            const label = document.createElement("label");
            label.htmlFor = `feature-${feature}`;
            label.textContent = feature.replaceAll("_", " ");

            const input = document.createElement("input");
            input.id = `feature-${feature}`;
            input.name = feature;
            input.type = "number";
            input.step = "any";
            input.required = true;
            input.placeholder = "Enter value";

            field.append(label, input);
            fieldsContainer.appendChild(field);
        });
    } catch (error) {
        fieldsContainer.textContent = error.message;
        errorMessage.textContent =
            "Check that Flask is running and /features is working.";
    }
}

async function loadMetrics() {
    const metricsContainer = document.getElementById("metrics");

    try {
        const response = await fetch("/metrics");
        if (!response.ok) throw new Error("Metrics unavailable");

        const data = await response.json();
        metricsContainer.replaceChildren();

        const entries = Object.entries(data.metrics || data);

        entries.forEach(([name, value]) => {
            if (typeof value !== "number" || !Number.isFinite(value)) return;

            const card = document.createElement("div");
            card.className = "metric";

            const label = document.createElement("span");
            label.className = "muted small";
            label.textContent = name.toUpperCase();

            const number = document.createElement("strong");
            number.textContent = value.toFixed(3);

            card.append(label, number);
            metricsContainer.appendChild(card);
        });

        if (!metricsContainer.children.length) {
            metricsContainer.textContent = "No metrics available.";
        }
    } catch {
        metricsContainer.textContent =
            "Add a /metrics route to display training metrics.";
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorMessage.textContent = "";
    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    try {
        const payload = {};

        fieldsContainer.querySelectorAll("input").forEach((input) => {
            payload[input.name] = Number(input.value);
        });

        if (Object.values(payload).some((value) => !Number.isFinite(value))) {
            throw new Error("Please enter valid numeric values.");
        }

        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || data.message || "Prediction failed.");
        }

        const prediction =
            data.prediction ?? data.predicted_ph ?? data.predicted_value;

        if (prediction === undefined || !Number.isFinite(Number(prediction))) {
            throw new Error("The API response did not contain a numeric prediction.");
        }

        document.getElementById("result-value").textContent =
            Number(prediction).toFixed(2);

        document.getElementById("result-message").textContent =
            "Prediction generated successfully. This is a model estimate, not a water safety certification.";
    } catch (error) {
        errorMessage.textContent = error.message;
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Next-Day pH →";
    }
});

loadFeatures();
loadMetrics();
