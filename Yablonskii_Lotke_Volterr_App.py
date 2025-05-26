# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tktooltip import ToolTip
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import symbols, Matrix, N, solve, Eq, diff, dsolve, Function, re
from sympy import Function, solveset
from matplotlib.figure import Figure
from sympy import symbols, ln, lambdify
import matplotlib.animation as animation

class ImprovedLVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Модель Лотки-Вольтерра")
        self.setup_ui()
        self.methods = {
            'Аналитический (SymPy)': self.analytical_solution,
            'Эйлера': self.euler_method,
            'Рунге-Кутта 2': self.rk2_method,
            'Рунге-Кутта 4': self.rk4_method
        }

        self.analytic_figures = [None, None, None]
        self.analytic_canvas = [None, None, None]

        self.main_plots_visible = True
        self.analytic_plots_visible = False

        self.phase_history = [[], []]
        self.ani = None

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        param_frame = ttk.LabelFrame(main_frame, text="Параметры и настройки")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.params = self.create_parameter_inputs(param_frame)
        
        control_frame = ttk.Frame(param_frame)
        control_frame.pack(pady=10)
        
        ttk.Label(control_frame, text="Метод решения:").grid(row=0, column=0, sticky=tk.W)
        
        self.method_var = tk.StringVar()
        methods = ['Аналитический (SymPy)', 'Эйлера', 'Рунге-Кутта 2', 'Рунге-Кутта 4']
        self.method_combobox = ttk.Combobox(control_frame, textvariable=self.method_var, values=methods, state="readonly")
        ToolTip(self.method_combobox, msg = "Выбирается численный метод")
        self.method_combobox.current(0)
        self.method_combobox.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Шаг времени:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.dt_entry = ttk.Entry(control_frame, width=10)
        ToolTip(self.dt_entry, msg = "Выбирается шаг по времени")

        self.dt_entry.insert(0, "0.01")
        self.dt_entry.grid(row=1, column=1, padx=5)

        ttk.Button(param_frame, text="Запустить моделирование", 
                  command=self.run_simulation).pack(pady=10)
        
        result_frame = ttk.Frame(main_frame)
        ToolTip(result_frame, msg = "результаты")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure1 = plt.Figure(figsize=(10, 5))
        self.figure2 = plt.Figure(figsize=(10, 5))
        self.canvas = [None, None]
        
        frame1 = ttk.Frame(result_frame)
        frame1.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas[0] = FigureCanvasTkAgg(self.figure1, master=frame1)
        self.canvas[0].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        frame2 = ttk.Frame(result_frame)
        frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas[1] = FigureCanvasTkAgg(self.figure2, master=frame2)
        self.canvas[1].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.info_text = tk.Text(result_frame, height=8, width=60)
        self.info_text.pack(pady=5, fill=tk.X)

        ttk.Button(param_frame, text="Стоп анимация", 
                  command=self.stop_animation).pack(pady=5)


    def create_parameter_inputs(self, parent):
        params = [
            ('α (рождаемость жертв)', 'alpha', 1.0),
            ('β (смертность жертв)', 'beta', 1.0),
            ('γ (смертность хищников)', 'gamma', 0.5),
            ('δ (эффективность охоты)', 'delta', 0.5),
            ('ε (конкуренция жертв)', 'epsilon', 0.0),
            ('ζ (конкуренция хищников)', 'zeta', 0.0),
            ('Начало. жертвы', 'x0', 1.0),
            ('Начало. хищники', 'y0', 0.25),
            ('Время моделирования', 'time', 500)
        ]
        
        entries = {}
        for i, (label, key, default) in enumerate(params):
            row = ttk.Frame(parent)
            ToolTip(row, msg = "Ввод начальных данных")
            
            row.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=10)
            entry.insert(0, str(default))
            entry.pack(side=tk.RIGHT)
            entries[key] = entry
        return entries


    def get_parameters(self):
        try:
            params = {key: float(entry.get()) for key, entry in self.params.items()}
            params['dt'] = float(self.dt_entry.get())
            params['method'] = self.method_var.get()
            return params
        except ValueError:
            return None

    def run_simulation(self):
        params = self.get_parameters()
        if not params:
            self.show_error("Некорректные параметры!")
            return
        
        self.stop_animation()

        if params['method'] == 'Аналитический (SymPy)':
            if params['epsilon'] != 0 or params['zeta'] != 0:
                self.show_error("Аналитическое решение только для ε=0 и ζ=0!")
                return
            params_dict = {
                symbols('alpha'): params['alpha'],
                symbols('beta'): params['beta'],
                symbols('gamma'): params['gamma'],
                symbols('delta'): params['delta']
            }
            self.plot_analytic_results(params_dict, params)
        else:
            t = np.linspace(0, params['time'], int(params['time']/params['dt']))
            method_func = self.methods[params['method']]
            sol = method_func(params, t)
            # Получение решения через odeint
            ode_sol = odeint(
                self.dsdt,
                [params['x0'], params['y0']],
                t,
                args=(
                    params['alpha'],
                    params['beta'],
                    params['gamma'],
                    params['delta'],
                    params['epsilon'],
                    params['zeta']
                )
            )
            self.solution = sol
            self.ode_solution = ode_sol

            residual = None
            if params['epsilon'] == 0 and params['zeta'] == 0:
                try:
                    x = sol[:, 0]
                    y = sol[:, 1]
                    if np.any(x <= 0) or np.any(y <= 0):
                        raise ValueError("Популяция достигла неположительного значения.")
                    C0 = (params['delta'] * params['x0'] 
                          - params['gamma'] * np.log(params['x0']) 
                          + params['beta'] * params['y0'] 
                          - params['alpha'] * np.log(params['y0']))
                    C = (params['delta'] * x 
                         - params['gamma'] * np.log(x) 
                         + params['beta'] * y 
                         - params['alpha'] * np.log(y))
                    residual = np.max(np.abs(sol - ode_sol))
                except Exception as e:
                    residual = None
                    self.show_error(f"Ошибка вычисления невязки: {str(e)}")
            
            self.animate_simulation(params, t)
            self.show_stationary_info(params, residual)

        self.show_stationary_info(params, residual if 'residual' in locals() else None)

    def animate_simulation(self, params, t):
        self.t_points = t
        self.current_step = 0

        self.figure1.clf()
        self.ax1 = self.figure1.add_subplot(111)
        # Линии для численного метода
        self.line1, = self.ax1.plot([], [], color='blue', label='Жертвы (числ)')
        self.line2, = self.ax1.plot([], [], color='orange', label='Хищники (числ)')
        # Линии для odeint
        self.line1_ode, = self.ax1.plot([], [], '--', color='blue', label='Жертвы (odeint)')
        self.line2_ode, = self.ax1.plot([], [], '--', color='orange', label='Хищники (odeint)')
        
        # Определение максимального значения для оси Y
        max_x_num = np.max(self.solution[:, 0])
        max_y_num = np.max(self.solution[:, 1])
        max_x_ode = np.max(self.ode_solution[:, 0])
        max_y_ode = np.max(self.ode_solution[:, 1])
        max_val = max(max_x_num, max_y_num, max_x_ode, max_y_ode)
        self.ax1.set_xlim(0, params['time'])
        self.ax1.set_ylim(0, max_val * 1.1)
        self.ax1.legend()

        self.figure2.clf()
        self.ax2 = self.figure2.add_subplot(111)
        self.phase_line, = self.ax2.plot([], [], 'g-', lw=1)
        self.ax2.set_xlim(0, params['x0']*10)
        self.ax2.set_ylim(0, params['y0']*20)
        self.ax2.set_title("Фазовый портрет системы")
        self.ax2.set_xlabel("Жертвы")
        self.ax2.set_ylabel("Хищники")

        self.phase_history = [[], []]

        self.ani = animation.FuncAnimation(
            self.figure1,
            self.update_animation,
            frames=len(t),
            fargs=(params,),
            interval=1,
            blit=False,
            repeat=False
        )
        
        self.canvas[0].draw()
        self.canvas[1].draw()

    def update_animation(self, frame, params):
        if frame == 0:
            self.phase_history = [[], []]
            return
        
        x_data = self.solution[:frame, 0]
        y_data = self.solution[:frame, 1]
        ode_x_data = self.ode_solution[:frame, 0]
        ode_y_data = self.ode_solution[:frame, 1]
        
        self.line1.set_data(self.t_points[:frame], x_data)
        self.line2.set_data(self.t_points[:frame], y_data)
        self.line1_ode.set_data(self.t_points[:frame], ode_x_data)
        self.line2_ode.set_data(self.t_points[:frame], ode_y_data)
        self.phase_line.set_data(x_data, y_data)
        
        self.ax1.relim()
        self.ax1.autoscale_view(scalex=False, scaley=True)
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=True, scaley=True)
        
        self.canvas[0].draw()
        self.canvas[1].draw()

        return self.line1, self.line2, self.line1_ode, self.line2_ode, self.phase_line

    def euler_step(self, params, frame):
        h = params['dt']
        x, y = self.solution[frame-1]
        dx = params['alpha']*x - params['beta']*x*y - params['epsilon']*x**2
        dy = -params['gamma']*y + params['delta']*x*y - params['zeta']*y**2
        self.solution[frame] = [x + dx*h, y + dy*h]

    def rk2_step(self, params, frame):
        h = params['dt']
        x, y = self.solution[frame-1]
        
        k1x = params['alpha']*x - params['beta']*x*y - params['epsilon']*x**2
        k1y = -params['gamma']*y + params['delta']*x*y - params['zeta']*y**2
        
        x_temp = x + k1x*h/2
        y_temp = y + k1y*h/2
        
        k2x = params['alpha']*x_temp - params['beta']*x_temp*y_temp - params['epsilon']*x_temp**2
        k2y = -params['gamma']*y_temp + params['delta']*x_temp*y_temp - params['zeta']*y_temp**2
        
        self.solution[frame] = [x + k2x*h, y + k2y*h]

    def rk4_step(self, params, frame):
        h = params['dt']
        x, y = self.solution[frame-1]
        
        k1x = params['alpha']*x - params['beta']*x*y - params['epsilon']*x**2
        k1y = -params['gamma']*y + params['delta']*x*y - params['zeta']*y**2
        
        k2x = params['alpha']*(x + k1x*h/2) - params['beta']*(x + k1x*h/2)*(y + k1y*h/2) - params['epsilon']*(x + k1x*h/2)**2
        k2y = -params['gamma']*(y + k1y*h/2) + params['delta']*(x + k1x*h/2)*(y + k1y*h/2) - params['zeta']*(y + k1y*h/2)**2
        
        k3x = params['alpha']*(x + k2x*h/2) - params['beta']*(x + k2x*h/2)*(y + k2y*h/2) - params['epsilon']*(x + k2x*h/2)**2
        k3y = -params['gamma']*(y + k2y*h/2) + params['delta']*(x + k2x*h/2)*(y + k2y*h/2) - params['zeta']*(y + k2y*h/2)**2
        
        k4x = params['alpha']*(x + k3x*h) - params['beta']*(x + k3x*h)*(y + k3y*h) - params['epsilon']*(x + k3x*h)**2
        k4y = -params['gamma']*(y + k3y*h) + params['delta']*(x + k3x*h)*(y + k3y*h) - params['zeta']*(y + k3y*h)**2
        
        self.solution[frame] = [
            x + (k1x + 2*k2x + 2*k3x + k4x)*h/6,
            y + (k1y + 2*k2y + 2*k3y + k4y)*h/6
        ]

    def stop_animation(self):
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None


    def toggle_plots(self, show_analytic):
        """Управление видимостью графиков"""
        if show_analytic:
            self.analytic_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            for frame in [self.canvas[0].get_tk_widget(), 
                        self.canvas[1].get_tk_widget()]:
                frame.pack_forget()
        else:
            self.analytic_frame.pack_forget()
            self.canvas[0].get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvas[1].get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def update_plots(self, t, solution, params):
        """Обновление только для численных методов"""
        if self.main_plots_visible:
            self.figure1.clf()
            self.figure2.clf()
                
            # График динамики
            ax1 = self.figure1.add_subplot(111)
            ax1.plot(t, solution[:, 0], label='Жертвы')
            ax1.plot(t, solution[:, 1], label='Хищники')
            ax1.set_title(f"Динамика популяций ({params['method']})")
            ax1.legend()

            # Фазовый портрет
            ax2 = self.figure2.add_subplot(111)
            ax2.plot(solution[:, 0], solution[:, 1], 'g')
            ax2.set_title("Фазовый портрет системы")
            ax2.set_xlabel("Жертвы")
            ax2.set_ylabel("Хищники")

            for canvas in self.canvas:
                canvas.draw()

    def show_stationary_info(self, params, residual=None):
        a, b, g, d, e, z = (params[k] for k in 
                            ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'])
        
        text = ""
        try:
            x, y = symbols('x y')
            eq1 = Eq(a*x - b*x*y - e*x**2, 0)
            eq2 = Eq(-g*y + d*x*y - z*y**2, 0)
            solutions = solve((eq1, eq2), (x, y))
            
            text = "Стационарные точки:\n"
            for sol in solutions:
                try:
                    x_val = complex(sol[0])
                    y_val = complex(sol[1])
                    
                    if x_val.imag != 0 or y_val.imag != 0 or x_val.real < 0 or y_val.real < 0:
                        text += f"({sol[0]:.2f}, {sol[1]:.2f}) - Нереализуема\n"
                        continue
                        
                    x_val = x_val.real
                    y_val = y_val.real
                    
                    text += f"({x_val:.2f}, {y_val:.2f})\n"
                    
                    J = Matrix([
                        [a - b*y_val - 2*e*x_val, -b*x_val],
                        [d*y_val, -g + d*x_val - 2*z*y_val]
                    ])
                    
                    eigenvals = J.eigenvals()
                    eigenvalues = [N(λ) for λ in eigenvals.keys()]
                    
                    classification = self.classify_stationary_point(eigenvalues)
                    text += f"Тип точки: {classification}\n"
                    text += f"Собственные значения: {eigenvalues}\n\n"
                    
                except Exception as ex:
                    text += f"Ошибка анализа точки: {str(ex)}\n"

            if residual is not None:
                text += f"\nМаксимальная невязка: {residual:.4e}\n"
            elif e == 0 and z == 0:
                text += "\nНевязка не может быть вычислена.\n"

        except Exception as e:
            text = f"Ошибка вычислений: {str(e)}"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)


    def classify_stationary_point(self, eigenvalues):
        λ1 = eigenvalues[0]
        λ2 = eigenvalues[1]
        
        # Проверка на комплексные значения
        if λ1.as_real_imag()[1] != 0 or λ2.as_real_imag()[1] != 0:
            re1 = λ1.as_real_imag()[0]
            re2 = λ2.as_real_imag()[0]
            
            if re1 < 0 and re2 < 0:
                return "Устойчивый фокус"
            elif re1 > 0 and re2 > 0:
                return "Неустойчивый фокус"
            else:
                return "Центр (нейтральная устойчивость)"
        
        # Действительные значения
        λ1 = float(re(λ1))
        λ2 = float(re(λ2))
        
        if λ1 * λ2 < 0:
            return "Седло"
        elif λ1 > 0 and λ2 > 0:
            return "Неустойчивый узел"
        elif λ1 < 0 and λ2 < 0:
            return "Устойчивый узел"
        else:
            return "Особый случай"

    
    def analytical_solution(self, params, t):
        if params['epsilon'] != 0 or params['zeta'] != 0:
            self.show_error("Аналитическое решение только для ε=0 и ζ=0!")
            return np.zeros((len(t), 2))

        try:
            # Создаем params_dict с символами в качестве ключей
            alpha_sym, beta_sym, gamma_sym, delta_sym = symbols('alpha beta gamma delta')
            params_dict = {
                alpha_sym: params['alpha'],
                beta_sym: params['beta'],
                gamma_sym: params['gamma'],
                delta_sym: params['delta']
            }
            
            # Построение интегральных кривых
            self.plot_analytic_results(params_dict, params)
            
            return np.zeros((len(t), 2))  # Возвращаем пустой массив
        
        except Exception as e:
            self.show_error(f"Ошибка аналитического решения: {str(e)}")
            return np.zeros((len(t), 2))

    def dsdt(self, s, t, *args):
        """Система дифференциальных уравнений"""
        params = {
            'alpha': args[0],
            'beta': args[1],
            'gamma': args[2],
            'delta': args[3],
            'epsilon': args[4],
            'zeta': args[5]
        }
        x, y = s
        dx = params['alpha']*x - params['beta']*x*y - params['epsilon']*x**2
        dy = -params['gamma']*y + params['delta']*x*y - params['zeta']*y**2
        return [dx, dy]

    def plot_analytic_results(self, params_dict, params):
        # Очистка предыдущих графиков
        self.figure1.clf()
        self.figure2.clf()

        try:
            # Интегральные кривые
            ax1 = self.figure1.add_subplot(111)
            X, Y, C = self.compute_contour_data(params_dict)
            ax1.contour(X, Y, C, levels=50, alpha=0.5)
            ax1.set_title("Интегральные кривые")

            # Векторное поле
            ax2 = self.figure2.add_subplot(111)
            X, Y, U, V = self.compute_vector_field(params)
            U1 = U / np.sqrt(U * U + V * V)
            V1 = V / np.sqrt(U * U + V * V)
            ax2.quiver(X, Y, U1, V1, scale=50, color="orange")
            solution = odeint(
                self.dsdt,
                y0=[params['x0'], params['y0']],
                t=np.linspace(0, 50, 1000),
                args=tuple(params.values())
            )
            ax2.plot(solution[:,0], solution[:,1], "skyblue")
            ax2.set_title("Векторное поле")

            # Обновление canvas
            self.canvas[0].draw()
            self.canvas[1].draw()

        except Exception as e:
            self.show_error(f"Ошибка построения графиков: {str(e)}")


    def compute_first_integral(self, params_dict):
        # Определяем все необходимые символы
        x, y = symbols('x y')
        alpha, beta, gamma, delta = symbols('alpha beta gamma delta')
        C1 = symbols("C1")
        
        # Используем params_dict с символами в качестве ключей
        expr = (
            params_dict[delta] * x 
            - params_dict[gamma] * ln(x) 
            + params_dict[beta] * y 
            - params_dict[alpha] * ln(y)
        )
        return Eq(expr, C1)


    def compute_contour_data(self, params_dict):
        # Определяем символы
        x, y = symbols('x y')
        alpha, beta, gamma, delta = symbols('alpha beta gamma delta')
        
        # Диапазоны для сетки
        x_range = np.arange(0.01, 7, 0.01)
        y_range = np.arange(0.01, 7, 0.01)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Получаем выражение для интеграла
        expr = self.compute_first_integral(params_dict).lhs
        # Подставляем параметры и создаем функцию
        cf = lambdify((x, y), expr.subs(params_dict), 'numpy')
        
        return X, Y, cf(X, Y)


    def compute_vector_field(self, params):
        x_range = np.arange(0.1, 7, 0.35)
        y_range = np.arange(0.1, 7, 0.35)
        X, Y = np.meshgrid(x_range, y_range)
        
        dx = params['alpha']*X - params['beta']*X*Y
        dy = -params['gamma']*Y + params['delta']*X*Y
        
        return X, Y, dx, dy


    def euler_method(self, params, t):
        def deriv(s, _):
            x, y = s
            dx = params['alpha']*x - params['beta']*x*y - params['epsilon']*x**2
            dy = -params['gamma']*y + params['delta']*x*y - params['zeta']*y**2
            return [dx, dy]
        
        sol = np.zeros((len(t), 2))
        sol[0] = [params['x0'], params['y0']]
        for i in range(1, len(t)):
            h = t[i] - t[i-1]
            sol[i] = sol[i-1] + h * np.array(deriv(sol[i-1], t[i-1]))
        return sol

    def rk2_method(self, params, t):
        def deriv(s, _):
            x, y = s
            dx = params['alpha']*x - params['beta']*x*y - params['epsilon']*x**2
            dy = -params['gamma']*y + params['delta']*x*y - params['zeta']*y**2
            return np.array([dx, dy])
        
        sol = np.zeros((len(t), 2))
        sol[0] = [params['x0'], params['y0']]
        for i in range(1, len(t)):
            h = t[i] - t[i-1]
            k1 = deriv(sol[i-1], t[i-1])
            k2 = deriv(sol[i-1] + k1*h/2, t[i-1] + h/2)
            sol[i] = sol[i-1] + h*k2
        return sol

    def rk4_method(self, params, t):
        def deriv(s, _):
            x, y = s
            dx = params['alpha']*x - params['beta']*x*y - params['epsilon']*x**2
            dy = -params['gamma']*y + params['delta']*x*y - params['zeta']*y**2
            return np.array([dx, dy])
        
        sol = np.zeros((len(t), 2))
        sol[0] = [params['x0'], params['y0']]
        for i in range(1, len(t)):
            h = t[i] - t[i-1]
            k1 = deriv(sol[i-1], t[i-1])
            k2 = deriv(sol[i-1] + k1*h/2, t[i-1] + h/2)
            k3 = deriv(sol[i-1] + k2*h/2, t[i-1] + h/2)
            k4 = deriv(sol[i-1] + k3*h, t[i-1] + h)
            sol[i] = sol[i-1] + (k1 + 2*k2 + 2*k3 + k4) * h / 6
        return sol

    def show_error(self, message):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "ОШИБКА: " + message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedLVApp(root)
    root.mainloop()