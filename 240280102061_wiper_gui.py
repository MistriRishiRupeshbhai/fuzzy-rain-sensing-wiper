import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Fuzzy variables

rain_intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'rain_intensity')
vehicle_speed = ctrl.Antecedent(np.arange(0, 151, 1), 'vehicle_speed')
wiper_speed = ctrl.Consequent(np.arange(0, 101, 1), 'wiper_speed')


# Membership functions

rain_intensity['dry'] = fuzz.trimf(rain_intensity.universe, [0, 0, 40])
rain_intensity['moderate'] = fuzz.trimf(rain_intensity.universe, [20, 50, 80])
rain_intensity['heavy'] = fuzz.trimf(rain_intensity.universe, [60, 100, 100])

vehicle_speed['slow'] = fuzz.trimf(vehicle_speed.universe, [0, 0, 60])
vehicle_speed['medium'] = fuzz.trimf(vehicle_speed.universe, [30, 75, 120])
vehicle_speed['fast'] = fuzz.trimf(vehicle_speed.universe, [90, 150, 150])

wiper_speed['off'] = fuzz.trimf(wiper_speed.universe, [0, 0, 20])
wiper_speed['intermittent'] = fuzz.trimf(wiper_speed.universe, [10, 30, 50])
wiper_speed['slow'] = fuzz.trimf(wiper_speed.universe, [35, 60, 80])
wiper_speed['fast'] = fuzz.trimf(wiper_speed.universe, [65, 100, 100])


# Fuzzy rules

rule1 = ctrl.Rule(rain_intensity['dry'], wiper_speed['off'])

rule2 = ctrl.Rule(rain_intensity['moderate'] & vehicle_speed['slow'], wiper_speed['intermittent'])
rule3 = ctrl.Rule(rain_intensity['moderate'] & vehicle_speed['medium'], wiper_speed['slow'])
rule4 = ctrl.Rule(rain_intensity['moderate'] & vehicle_speed['fast'], wiper_speed['slow'])

rule5 = ctrl.Rule(rain_intensity['heavy'] & vehicle_speed['slow'], wiper_speed['slow'])
rule6 = ctrl.Rule(rain_intensity['heavy'] & vehicle_speed['medium'], wiper_speed['fast'])
rule7 = ctrl.Rule(rain_intensity['heavy'] & vehicle_speed['fast'], wiper_speed['fast'])

wiper_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])


# Decision function

def get_decision(value):
    if value < 15:
        return "WIPERS OFF"
    elif value < 45:
        return "INTERMITTENT WIPE"
    elif value < 80:
        return "SLOW CONTINUOUS WIPE"
    else:
        return "FAST CONTINUOUS WIPE"


# Membership calculation

def calculate_memberships(variable, value):
    result = {}

    for name, term in variable.terms.items():
        result[name] = fuzz.interp_membership(
            variable.universe,
            term.mf,
            value
        )

    return result


# Main window

root = tk.Tk()
root.title("Rain Sensing Wiper Controller")
root.geometry("1250x800")
root.minsize(1050, 700)


# Styles

style = ttk.Style()

try:
    style.theme_use("clam")
except:
    pass

style.configure(
    "Title.TLabel",
    font=("Arial", 22, "bold")
)

style.configure(
    "Subtitle.TLabel",
    font=("Arial", 11)
)

style.configure(
    "Result.TLabel",
    font=("Arial", 18, "bold")
)


# Title

title = ttk.Label(
    root,
    text="RAIN SENSING WIPER CONTROLLER",
    style="Title.TLabel"
)

title.pack(pady=(15, 2))

subtitle = ttk.Label(
    root,
    text="Mamdani Fuzzy Logic Controller",
    style="Subtitle.TLabel"
)

subtitle.pack(pady=(0, 10))


# Main layout

main_frame = ttk.Frame(root)
main_frame.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=10
)


# Input panel

input_frame = ttk.LabelFrame(
    main_frame,
    text="INPUT PARAMETERS",
    padding=15
)

input_frame.pack(
    side="left",
    fill="y",
    padx=(0, 10)
)


ttk.Label(
    input_frame,
    text="Rain Intensity (%)"
).pack(anchor="w", pady=(5, 3))

rain_entry = ttk.Entry(
    input_frame,
    width=25
)

rain_entry.pack(pady=(0, 15))
rain_entry.insert(0, "35")


ttk.Label(
    input_frame,
    text="Vehicle Speed (km/h)"
).pack(anchor="w", pady=(5, 3))

speed_entry = ttk.Entry(
    input_frame,
    width=25
)

speed_entry.pack(pady=(0, 20))
speed_entry.insert(0, "95")


# Result panel

result_frame = ttk.LabelFrame(
    main_frame,
    text="WIPER DECISION",
    padding=15
)

result_frame.pack(
    side="left",
    fill="y",
    padx=(0, 10)
)


ttk.Label(
    result_frame,
    text="Wiper Speed"
).pack(pady=(10, 2))

wiper_result = ttk.Label(
    result_frame,
    text="-- %",
    style="Result.TLabel"
)

wiper_result.pack(pady=(0, 20))


ttk.Label(
    result_frame,
    text="Decision"
).pack(pady=(5, 2))

decision_result = ttk.Label(
    result_frame,
    text="WAITING",
    style="Result.TLabel"
)

decision_result.pack(pady=(0, 20))


ttk.Label(
    result_frame,
    text="System Status"
).pack(pady=(5, 2))

status_result = ttk.Label(
    result_frame,
    text="READY"
)

status_result.pack(pady=(0, 20))


# Fuzzification panel

fuzzy_frame = ttk.LabelFrame(
    result_frame,
    text="Fuzzification",
    padding=10
)

fuzzy_frame.pack(
    fill="both",
    expand=True
)

fuzzy_text = tk.Text(
    fuzzy_frame,
    width=30,
    height=18,
    font=("Consolas", 9),
    wrap="word"
)

fuzzy_text.pack(
    fill="both",
    expand=True
)


# Graph panel

graph_frame = ttk.LabelFrame(
    main_frame,
    text="WIPER SPEED MEMBERSHIP",
    padding=5
)

graph_frame.pack(
    side="left",
    fill="both",
    expand=True
)


fig, ax = plt.subplots(
    figsize=(7, 5),
    dpi=90
)


def draw_graph(wiper_output=None):

    ax.clear()

    for term_name, term in wiper_speed.terms.items():
        ax.plot(
            wiper_speed.universe,
            term.mf,
            linewidth=2,
            label=term_name.capitalize()
        )

    if wiper_output is not None:

        ax.axvline(
            wiper_output,
            linestyle="--",
            linewidth=2,
            label=f"Wiper Speed = {wiper_output:.2f}%"
        )

        ax.scatter(
            [wiper_output],
            [0],
            s=70
        )

    ax.set_title(
        "Wiper Speed Membership Functions",
        fontweight="bold"
    )

    ax.set_xlabel("Wiper Speed (%)")
    ax.set_ylabel("Membership Degree (\u03bc)")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.05)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    canvas.draw()


canvas = FigureCanvasTkAgg(
    fig,
    master=graph_frame
)

canvas.get_tk_widget().pack(
    fill="both",
    expand=True
)

draw_graph()


# Calculate function

def calculate():

    try:
        rain_value = float(
            rain_entry.get()
        )

        speed_value = float(
            speed_entry.get()
        )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numerical values."
        )

        return


    if not 0 <= rain_value <= 100:

        messagebox.showerror(
            "Invalid Rain Intensity",
            "Rain intensity must be between 0 and 100 %."
        )

        return


    if not 0 <= speed_value <= 150:

        messagebox.showerror(
            "Invalid Speed",
            "Vehicle speed must be between 0 and 150 km/h."
        )

        return


    rain_memberships = calculate_memberships(
        rain_intensity,
        rain_value
    )

    speed_memberships = calculate_memberships(
        vehicle_speed,
        speed_value
    )


    simulation = ctrl.ControlSystemSimulation(
        wiper_ctrl
    )

    simulation.input['rain_intensity'] = rain_value
    simulation.input['vehicle_speed'] = speed_value


    try:
        simulation.compute()

        wiper_output = simulation.output[
            'wiper_speed'
        ]

    except Exception as error:

        messagebox.showerror(
            "Controller Error",
            f"Unable to calculate wiper speed.\n\n{error}"
        )

        return


    decision = get_decision(
        wiper_output
    )


    wiper_result.config(
        text=f"{wiper_output:.2f} %"
    )

    decision_result.config(
        text=decision
    )

    status_result.config(
        text="WIPER SPEED CALCULATED"
    )


    # Display fuzzification

    fuzzy_text.delete(
        "1.0",
        tk.END
    )

    fuzzy_text.insert(
        tk.END,
        "INPUT VALUES\n"
    )

    fuzzy_text.insert(
        tk.END,
        "-" * 30 + "\n"
    )

    fuzzy_text.insert(
        tk.END,
        f"Rain Intensity : {rain_value:.2f} %\n"
    )

    fuzzy_text.insert(
        tk.END,
        f"Vehicle Speed  : {speed_value:.2f} km/h\n\n"
    )


    fuzzy_text.insert(
        tk.END,
        "RAIN INTENSITY MEMBERSHIP\n"
    )

    fuzzy_text.insert(
        tk.END,
        "-" * 30 + "\n"
    )

    for name, value in rain_memberships.items():

        fuzzy_text.insert(
            tk.END,
            f"{name.capitalize():12s}: {value:.3f}\n"
        )


    fuzzy_text.insert(
        tk.END,
        "\nVEHICLE SPEED MEMBERSHIP\n"
    )

    fuzzy_text.insert(
        tk.END,
        "-" * 30 + "\n"
    )

    for name, value in speed_memberships.items():

        fuzzy_text.insert(
            tk.END,
            f"{name.capitalize():12s}: {value:.3f}\n"
        )


    fuzzy_text.insert(
        tk.END,
        "\nFINAL OUTPUT\n"
    )

    fuzzy_text.insert(
        tk.END,
        "-" * 30 + "\n"
    )

    fuzzy_text.insert(
        tk.END,
        f"Wiper Speed : {wiper_output:.2f}%\n"
    )

    fuzzy_text.insert(
        tk.END,
        f"Decision    : {decision}\n"
    )


    draw_graph(wiper_output)


# Reset function

def reset():

    rain_entry.delete(0, tk.END)
    rain_entry.insert(0, "35")

    speed_entry.delete(0, tk.END)
    speed_entry.insert(0, "95")

    wiper_result.config(text="-- %")
    decision_result.config(text="WAITING")
    status_result.config(text="READY")

    fuzzy_text.delete(
        "1.0",
        tk.END
    )

    draw_graph()


# Buttons

calculate_button = ttk.Button(
    input_frame,
    text="CALCULATE WIPER SPEED",
    command=calculate
)

calculate_button.pack(
    fill="x",
    pady=5
)

reset_button = ttk.Button(
    input_frame,
    text="RESET",
    command=reset
)

reset_button.pack(
    fill="x",
    pady=5
)


# Footer

footer = ttk.Label(
    root,
    text="Phase 6 - User Interface | Mamdani Fuzzy Inference System",
    anchor="center"
)

footer.pack(
    fill="x",
    pady=(0, 8)
)


root.mainloop()
