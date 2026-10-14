# Starting with uv

## 🧠 Concept: What is a Virtual Environment?

Before we install anything, it is critical to understand **why** we do this.

Imagine you are a chef with **one single kitchen** (your computer).

* **Without Virtual Environments:** You cook a spicy curry (Project A) and a delicate dessert (Project B) on the same counter at the same time. The spices spill, the flavors mix, and both dishes are ruined. In Python, this is like Project A needing `pandas version 1` and Project B needing `pandas version 2`. If you install them globally, they conflict and break.
* **With Virtual Environments:** You buy a separate, sealed **lunchbox** for every recipe. You put the specific ingredients for Project A in Box A, and ingredients for Project B in Box B. Nothing mixes.

**The Golden Rule:** Never install libraries (like numpy or pandas) globally on your computer. Always create a virtual environment (a "lunchbox") for every project.

**The Good News:** The tool we are about to use, `uv`, creates and manages these boxes for you automatically.

---

## 🚀 Step 1: Install `uv`

Run the command for your system. `uv` will handle Python installation for you.

* **🪟 Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```


* **🍎 macOS / 🐧 Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```



> *Tip: If your terminal says "uv not found" after this, close and reopen the terminal window.*

🔗 **Official Docs:** [uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

---

## 📁 Step 2: Create Your Project

Go to the folder where you keep your work and create a new project.

```bash
# 1. Create a folder
mkdir data_science_project
cd data_science_project

# 2. Initialize the project
uv init
```

*`uv` has now created a `main.py` file and a `pyproject.toml` file (which tracks your settings).*

🔗 **Official Docs:** [uv Projects Guide](https://docs.astral.sh/uv/guides/projects/)

---

## 🪄 Step 3: Run the Code (Automatic Python Setup)

This is the magic step. We will run the default script. **If you don't have Python installed, `uv` will download and install it for you automatically right now.**

```bash
uv run main.py
```

**You should see output similar to this:**

```text
Using CPython 3.12.3
Creating virtual environment at: .venv
Hello from data_science_project!
```

**What just happened?**

1. `uv` saw you needed Python and downloaded a standalone version (isolated from your system).
2. It created a **Virtual Environment** (the folder named `.venv`).
3. It ran the script inside that isolated environment.

🔗 **Official Docs:** [uv Python Version Management](https://docs.astral.sh/uv/guides/install-python/)

---

## 📦 Step 4: Add Libraries

Now we add **NumPy** (for math) and **Scikit-learn** (for machine learning).
*Note: We install it as `scikit-learn`, but in Python code we import it as `sklearn`.*

```bash
uv add numpy scikit-learn
```

*`uv` is now installing these libraries into your isolated `.venv` folder.*

**✅ Verification:**
To see exactly what libraries are installed, run this:

```bash
uv run pip list
```

---

## 🧪 Step 5: Run a Machine Learning Script

Let's replace the default `main.py` with a real script.

1. Delete the code inside `main.py`.
2. Paste this code instead:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Create some dummy data (Training Data)
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([2, 4, 6, 8, 10])           # Targets (y = 2x)

# 2. Create and train a model
model = LinearRegression()
model.fit(X, y)

# 3. Predict the next value
prediction = model.predict([[6]])

print(f"✅ Model trained successfully!")
print(f"Prediction for input 6 (should be 12): {prediction[0]:.2f}")
```

**Run it:**

```bash
uv run main.py
```

---

## 🛠️ Step 6: Configure Your IDE

Your code works in the terminal, but for your IDE (VS Code, PyCharm, etc.) to understand your code, give you auto-complete, and check for errors, **you must point it to the Python interpreter inside your virtual environment.**

You need to tell your IDE to use the specific executable file located here:

* **🪟 Windows:** `data_science_project\.venv\Scripts\python.exe`
* **🍎 macOS / 🐧 Linux:** `data_science_project/.venv/bin/python` (or `python3`)

### **Visual Studio Code (VS Code)**

1. Open your project folder (`data_science_project`) in VS Code.
2. Open `main.py`.
3. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac).
4. Type and select: **Python: Select Interpreter**.
5. Select **"Enter interpreter path..."** -> **"Find..."**.
6. Navigate into your `.venv` folder and select the `python` (or `python.exe`) file described above.

### **PyCharm**

1. Open PyCharm and select **Open**, then choose your `data_science_project` folder.
2. Go to **Settings** (Windows) or **PyCharm** > **Settings** (Mac).
3. Navigate to **Project: data_science_project** > **Python Interpreter**.
4. Click **Add Interpreter** > **Local...** > **Virtualenv Environment**.
5. Select **Existing environment**.
6. Click the `...` (browse) button and navigate to select the `python` (or `python.exe`) file inside your `.venv` folder described above.
