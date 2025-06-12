# Birds Flocking Animation (2D & 3D, Vectorized Boids Algorithm)

This project demonstrates interactive 2D and 3D bird flocking animations based on the classic **Boids algorithm**—a well-known model for simulating the collective movement of flocks, herds, or swarms. Unlike most implementations, **this project uses a fully vectorized approach with NumPy**, avoiding for-loops and OOP for high performance and clarity.

Both 2D and 3D visualizations are provided, built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/), allowing users to experiment with the parameters of flocking behavior in real time.

---

## :bird: Live Demos

You can run the interactive apps directly from your browser.  
**Tip:** For best performance, use the Cloud Run location closer to you.

**2D Version:**  
- [Los Angeles (US West)](https://birds-animation-2d-los-angeles-932053663419.us-west2.run.app)  
- [Montreal (Canada East)](https://birds-animation-2d-montreal-932053663419.northamerica-northeast1.run.app)

**3D Version:**  
- [Los Angeles (US West)](https://birds-animation-3d-los-angeles-810457048638.us-west2.run.app)  
- [Montreal (Canada East)](https://birds-animation-3d-montreal-810457048638.northamerica-northeast1.run.app/)

> **Note:** Cloud Run has some latency. For the smoothest experience, lower the number of birds or increase the animation frame duration if needed.

---


## 🧠 How the Boids Algorithm Works

Each bird follows three simple rules based on nearby flockmates:

1. **Separation (Avoid):** Steer away to avoid overcrowding.
2. **Alignment:** Match velocity with neighbors.
3. **Cohesion (Centering):** Move toward the flock’s center.

These behaviors are modulated by adjustable weights and a “field of vision” distance.

---

## :star: Features

- **Real-time interactive animation** of bird flocking in 2D and 3D.
- Fully **vectorized NumPy** implementation—no for-loops or OOP!
- Interactive parameter controls:
  - Number of birds
  - Flocking rule weights (avoidance, alignment, centering)
  - Field of vision
  - Speed range
  - Frame duration (animation speed)
- Pause, play, and reset buttons.
- 3D version features interactive camera and cubic bounding box.

---

## :zap: Why Vectorized?

Traditional Boids code iterates through each bird and its neighbors using for-loops, often with object-oriented classes.  
**This project replaces all loops with vectorized matrix operations** (NumPy), making it more efficient and concise, and showcasing a different approach to agent-based simulation.

---

## :file_folder: Project Structure
- `birds_2d.py`
- `birds_3d.py`

---

## 🚀 Running Locally

1. **Install dependencies:**

    ```bash
    pip install dash plotly numpy
    ```

2. **Run the 2D simulation:**

    ```bash
    python birds_2d.py
    ```

   **Or run the 3D simulation:**

    ```bash
    python birds_3d.py
    ```

3. Point your browser to [http://localhost:8050](http://localhost:8050).

---


## 🛠️ Customization

- Tweak the flocking behavior in real time using the UI sliders.
- Change visual styles (marker symbols, colors, sizes) by editing constants at the top of each `.py` file.

---

## 📚 References

- Reynolds, C. W. (1987). *Flocks, Herds, and Schools: A Distributed Behavioral Model*.
- [Boids Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Boids)
- [Plotly Dash Documentation](https://dash.plotly.com/)

---

## 🙌 Credits

If you use or extend this project, your support is appreciated—feel free to star or fork!

---

## 📝 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute!

