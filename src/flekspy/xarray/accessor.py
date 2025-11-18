import xarray as xr
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata
from flekspy.util.safe_eval import safe_eval
from flekspy.util.utilities import get_unit
from flekspy.plot.streamplot import streamplot
import yt

@xr.register_dataset_accessor("fleks")
class FleksAccessor:
    def __init__(self, ds):
        self._obj = ds

    def evaluate_expression(self, expression: str, unit: str = "planet"):
        r"""
        Evaluates the variable expression and return the result of an YTArray.

        Args:
            expression (str): Python codes to be executed
            Example: expression = "np.log({rhos0}+{rhos1})"
        """
        if "{" not in expression:
            return self.get_variable(expression, unit)

        eval_context = {"np": np}

        def repl(match):
            var_name = match.group(1)
            eval_context[var_name] = self.get_variable(var_name, unit)
            return var_name

        expression_for_eval = re.sub(r"\{(.*?)\}", repl, expression)
        return safe_eval(expression_for_eval, eval_context)

    def get_variable(self, var, unit="planet"):
        r"""
        Return raw variables or calculate derived variables.

        Args:
            var (str): variable name

        Return: YTArray
        """
        ytarr = None
        if var in self._obj.data_vars:
            varUnit = get_unit(var, unit)
            ytarr = yt.YTArray(self._obj[var].values, varUnit)
        else:
            var = var.lower()
            expression = None
            if var == "b":
                expression = "np.sqrt({Bx}**2+{By}**2+{Bz}**2)"
                varUnit = get_unit("b", unit)
            elif var == "bb":
                expression = "{Bx}**2+{By}**2+{Bz}**2"
                varUnit = get_unit("b", unit) + "**2"
            elif var[0:2] == "ps":
                ss = var[2:3]
                expression = (
                    "({pxxs" + ss + "}+" + "{pyys" + ss + "}+" + "{pzzs" + ss + "})/3"
                )
                varUnit = get_unit("p", unit)
            elif var == "pb":
                coef = 0.5 / (yt.units.mu_0.value)
                ytarr = coef * self.get_variable("bb", "si")
                ytarr = yt.YTArray(ytarr, "Pa")
                varUnit = get_unit("p", unit)
            elif var == "pbeta":
                ytarr = (
                    self.get_variable("ps0", unit) + self.get_variable("ps1", unit)
                ) / self.get_variable("pb", unit)
                varUnit = "dimensionless"
            elif var == "calfven":
                ytarr = self.get_variable("b", "si") / np.sqrt(
                    yt.units.mu_0.value * self.get_variable("rhos1", "si")
                )
                ytarr = yt.YTArray(ytarr, "m/s")
                varUnit = get_unit("u", unit)

            if expression is not None:
                ytarr = self.evaluate_expression(expression, unit)
                if not isinstance(ytarr, yt.units.yt_array.YTArray):
                    varUnit = "dimensionless"
                    ytarr = yt.YTArray(ytarr, varUnit)

        if ytarr is None:
            raise KeyError(f"Variable '{var}' not found in dataset.")

        return ytarr if str(ytarr.units) == "dimensionless" else ytarr.in_units(varUnit)

    def analyze_variable_string(self, var: str):
        vMin = None
        vMax = None

        varName = var
        if varName.find(">") > 0:
            varName = varName[: varName.find(">")]

        if varName.find("<") > 0:
            varName = varName[: varName.find("<")]

        if var.find(">") > 0:
            tmpVar = var[var.find(">") + 2 :]
            p1 = tmpVar.find(")")
            vMin = float(tmpVar[:p1])

        if var.find("<") > 0:
            tmpVar = var[var.find("<") + 2 :]
            p1 = tmpVar.find(")")
            vMax = float(tmpVar[:p1])

        return varName, vMin, vMax

    def plot(
        self,
        vars,
        xlim=None,
        ylim=None,
        unit: str = "planet",
        nlevels: int = 200,
        cmap: str = "turbo",
        figsize=(10, 6),
        f=None,
        axes=None,
        pcolor=False,
        logscale=False,
        addgrid=False,
        bottomline=10,
        showcolorbar: bool = True,
        *args,
        **kwargs,
    ):
        if isinstance(vars, str):
            vars = vars.split()

        nvar = len(vars)

        varNames = []
        varMin = []
        varMax = []
        for var in vars:
            vname, vmin, vmax = self.analyze_variable_string(var)
            varNames.append(vname)
            varMin.append(vmin)
            varMax.append(vmax)
        if f is None:
            f, axes = plt.subplots(nvar, 1, figsize=figsize, layout="constrained")
        axes = np.array(axes)
        axes = axes.reshape(-1)

        for isub, ax in zip(range(nvar), axes):
            ytVar = self.evaluate_expression(varNames[isub], unit)
            v = ytVar
            varUnit = "dimensionless"
            if isinstance(ytVar, yt.units.yt_array.YTArray):
                v = ytVar.value
                varUnit = str(ytVar.units)

            vmin = v.min() if varMin[isub] is None else varMin[isub]
            vmax = v.max() if varMax[isub] is None else varMax[isub]

            if logscale:
                v = np.log10(v)

            levels = np.linspace(vmin, vmax, nlevels)
            coords = list(self._obj.coords.keys())
            if len(coords) == 1:
                x = self._obj[coords[0]]
                ax.plot(x, v)
                ax.set_xlabel(x.name)
                ax.set_title(varNames[isub])
                continue

            x, y = self._obj[coords[0]], self._obj[coords[1]]

            if self._obj.attrs.get("gencoord", False):
                if pcolor or abs(vmin - vmax) < 1e-20 * abs(vmax):
                    cs = ax.tripcolor(x.values, y.values, v.T, cmap=cmap, *args, **kwargs)
                else:
                    cs = ax.tricontourf(
                        x.values, y.values, v.T, levels=levels, cmap=cmap, extend="both", *args, **kwargs
                    )
            else:
                if pcolor or abs(vmin - vmax) < 1e-20 * abs(vmax):
                    cs = ax.pcolormesh(x.values, y.values, v.T, cmap=cmap, *args, **kwargs)
                else:
                    cs = ax.contourf(
                        x.values, y.values, v.T, levels=levels, cmap=cmap, extend="both", *args, **kwargs
                    )
            if addgrid:
                if self._obj.attrs.get("gencoord", False):
                    gx, gy = x.values, y.values
                else:
                    gg = np.meshgrid(x.values, y.values)
                    gx, gy = np.reshape(gg[0], -1), np.reshape(gg[1], -1)
                ax.plot(gx, gy, "x")

            if showcolorbar:
                cb = f.colorbar(cs, ax=ax, pad=0.01)
                cb.formatter.set_powerlimits((0, 0))

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(x.name)
            ax.set_ylabel(y.name)
            title = varNames[isub]
            if varUnit != "dimensionless":
                title += f" [{varUnit}]"
            if logscale:
                title = f"$log_{{10}}$({title})"
            ax.set_title(title)

        if "cut_norm" in self._obj.attrs and "cut_loc" in self._obj.attrs:
            plt.figtext(0.01, 0.01, f"Plots at {self._obj.attrs['cut_norm']} = {self._obj.attrs['cut_loc']}", ha="left")

        return f, axes
