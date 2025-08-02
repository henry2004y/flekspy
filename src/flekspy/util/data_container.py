from copy import deepcopy

import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

from flekspy.util.utilities import get_unit
from flekspy.plot.streamplot import streamplot


def compare(d1, d2):
    header = ("var", "min|d1-d2|", "max|d1-d2|", "mean|d1-d2|", "mean|d1|", "mean|d2|")
    s = "{:8}   " + "{:18}" * 5
    print(s.format(*header))
    for var in d1.vars:
        dif = abs(d1.data[var] - d2.data[var])
        l = (
            var,
            float(dif.min()),
            float(dif.max()),
            float(dif.mean()),
            float(d1.data[var].mean()),
            float(d2.data[var].mean()),
        )
        s = "{:10}" + "{:+.6e},   " * 5
        print(s.format(*l))


class DataContainer(object):
    def __init__(
        self,
        dataSets,
        x,
        y,
        z,
        xlabel,
        ylabel,
        zlabel,
        step=-1,
        time=-1,
        gencoord=False,
        filename="",
    ):
        self.data = dataSets
        self.x = x
        self.y = y
        self.z = z
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

        self.vars = [x for x in self.data.keys()]
        self.range = [[x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]]]
        self.dimensions = self.data[self.vars[0]].shape
        self.nstep = step
        self.time = time
        self.filename = filename
        self.gencoord = gencoord

    def __repr__(self) -> str:
        str = (
            f"variables : {self.vars}\n"
            f"data range: {self.range}\n"
            f"dimension : {self.dimensions}\n"
        )

        return str

    def add_bottom_line(self, f, verbose):
        if verbose > 0:
            s = "time = " + str(self.time)
        if verbose > 1:
            s += " " * 5 + "nstep = " + str(self.nstep)
        if verbose > 2:
            s += " " * 20 + self.filename

        if verbose > 0:
            f.text(0.01, 0.01, s)

    def analyze_variable_string(self, var: str):
        r"""
        Parses the input string and return the plot variable and plot range.

        Args:
            var (str): string of variables to be parsed
            Example: var = "{bb}<(-10)>(-9.8)"

        Return: a tuple of the variable name, variable min and max.
        """
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

    def evaluate_expression(self, expression: str, unit: str = "planet"):
        r"""
        Evaluates the variable expression and return the result of an YTArray.

        Args:
            expression (str): Python codes to be executed
            Example: expression = "np.log({rhos0}+{rhos1})"
        """
        import re

        if '{' not in expression:
            return self.get_variable(expression, unit)

        # A dictionary to store the variables for eval.
        eval_context = {'np': np}

        def repl(match):
            var_name = match.group(1)
            # Add the variable to the context for eval.
            # Use a name that is valid for a python identifier.
            eval_context[var_name] = self.get_variable(var_name, unit)
            return var_name

        # Replace `{var}` with `var` and populate `eval_context`.
        expression_for_eval = re.sub(r'\{(.*?)\}', repl, expression)

        # Evaluate the expression in the prepared context.
        # We provide an empty `__builtins__` to restrict access to other built-in functions for security.
        return eval(expression_for_eval, {"__builtins__": {}}, eval_context)

    def add_variable(self, name, val):
        r"""
        Adds a variable to the dataset.

        Args:
            name (str): The name of the variable to be added.

            val: array-like structure
                Values of the variable stored in an array
        """

        if type(val) != yt.units.yt_array.YTArray:
            val = yt.YTArray(val, "dimensionless")

        if name not in self.data.keys():
            self.vars.append(name)
        self.data[name] = val

    def get_variable(self, var, unit="planet"):
        r"""
        Return raw variables or calculate derived variables.

        Args:
            var (str): variable name

        Return: YTArray
        """

        ytarr = None
        if var in self.data.keys():
            varUnit = get_unit(var, unit)
            ytarr = self.data[var]
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
                ytarr = yt.YTArray(ytarr.value, "Pa")
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
                ytarr = yt.YTArray(ytarr.value, "m/s")
                varUnit = get_unit("u", unit)

            if expression != None:
                ytarr = self.evaluate_expression(expression, unit)
                if type(ytarr) != yt.units.yt_array.YTArray:
                    varUnit = "dimensionless"
                    ytarr = yt.YTArray(ytarr, varUnit)

        if ytarr is None:
            raise KeyError(f"Variable '{var}' not found in dataset.")

        return ytarr if str(ytarr.units) == "dimensionless" else ytarr.in_units(varUnit)


class DataContainer3D(DataContainer):
    r"""
    A class handles 3D box data sets.
    """

    def __init__(
        self, dataSets, x, y, z, xlabel="X", ylabel="Y", zlabel="Z", *args, **kwargs
    ):
        r"""
        Args:
            dataSets (dict): The keys are variable names, and the dictionary values are YTArray.

            x/y/z: A 1D YTArray
        """
        super(DataContainer3D, self).__init__(
            dataSets, x, y, z, xlabel, ylabel, zlabel, *args, **kwargs
        )

    def get_slice(self, norm, cut_loc) -> "DataContainer2D":
        """Get a 2D slice from the 3D box data.

        Args:
            norm (str): The normal direction of the slice from "x", "y" or "z"

            cur_loc (float): The position of slicing.

        Return: DataContainer2D
        """

        axDir = {"X": 0, "Y": 1, "Z": 2}
        idir = axDir[norm.upper()]

        axes = [self.x, self.y, self.z]

        axSlice = axes[idir]

        for iSlice in range(axSlice.size):
            if axSlice[iSlice] > cut_loc:
                break

        dataSets = {}
        for varname, val in self.data.items():
            if idir == 0:
                arr = val[iSlice, :, :]
            elif idir == 1:
                arr = val[:, iSlice, :]
            elif idir == 2:
                arr = val[:, :, iSlice]
            dataSets[varname] = np.squeeze(arr)

        axLabes = {0: ("Y", "Z"), 1: ("X", "Z"), 2: ("X", "Y")}
        ax = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

        return DataContainer2D(
            dataSets,
            axes[ax[idir][0]],
            axes[ax[idir][1]],
            axLabes[idir][0],
            axLabes[idir][1],
            norm,
            cut_loc,
        )


class DataContainer2D(DataContainer):
    r"""
    A class handles 2D Cartesian data.
    """

    def __init__(
        self,
        dataSets,
        x,
        y,
        xlabel,
        ylabel,
        cut_norm=None,
        cut_loc=None,
        *args,
        **kwargs,
    ):
        r"""
        Args:
            dataSets (dict): The keys are variable names, and the values are YTArrays.

            x/y: 1D YTArray

            xlabel/ylabel (str): x, y dimension labels

            cut_norm (str): "x", "y" or "z"

            cut_loc (float): cut_norm and cut_loc are used to record the position of slice if this 2D data set is obtained from a 3D box.
        """

        zlabel = None
        z = (0, 0)
        super(DataContainer2D, self).__init__(
            dataSets, x, y, z, xlabel, ylabel, zlabel, *args, **kwargs
        )

        self.cut_norm = cut_norm
        self.cut_loc = cut_loc

    def __sub__(self, other):
        dif = deepcopy(self)
        for var in dif.vars:
            dif.data[var] -= other.data[var]
        return dif

    def plot(
        self,
        vars,
        xlim=None,
        ylim=None,
        unit: str = "planet",
        nlevels: int = 200,
        cmap: str = "turbo",
        figsize=(10, 6),
        f = None,
        axes = None,
        pcolor=False,
        logscale=False,
        addgrid=False,
        bottomline=10,
        showcolorbar: bool = True,
        *args,
        **kwargs,
    ):
        r"""
        2D plots.

        Args:
            vars (str): ploting variables and ploting range.
            Example: vars = "Bx<(50)>(-50) By (np.log(2*{rhos0}))>(-5)"

            xlim/ylim: A list/tuple contains the x- y-axis range

            unit (str): "planet" or "si"

            nlevels (int): Number of the contour levels. Default 200.

            cmap (str): color map type from Matplotlib

            figsize (tuple): size of figure. Default (10, 6).

            logscale (bool): True to scale the variable in log.

        Examples:
            >>> f, axes = dc.contour("Bx<(50)>(-50) By (np.log(2*{rhos0}))>(-5)", xlim=[-40,-5])
        """

        if type(vars) == str:
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
        axes = np.array(axes)  # in case nrows == ncols == 1
        axes = axes.reshape(-1)

        for isub, ax in zip(range(nvar), axes):
            ytVar = self.evaluate_expression(varNames[isub], unit)
            v = ytVar
            varUnit = "dimensionless"
            if type(ytVar) == yt.units.yt_array.YTArray:
                v = ytVar.value
                varUnit = str(ytVar.units)

            vmin = v.min() if varMin[isub] == None else varMin[isub]
            vmax = v.max() if varMax[isub] == None else varMax[isub]

            if logscale:
                v = np.log10(v)

            levels = np.linspace(vmin, vmax, nlevels)
            if self.gencoord:
                if pcolor or abs(vmin - vmax) < 1e-20 * abs(vmax):
                    cs = ax.tripcolor(
                        self.x.value, self.y.value, v.T, cmap=cmap, *args, **kwargs
                    )
                else:
                    cs = ax.tricontourf(
                        self.x.value,
                        self.y.value,
                        v.T,
                        levels=levels,
                        cmap=cmap,
                        extend="both",
                        *args,
                        **kwargs,
                    )
            else:
                if pcolor or abs(vmin - vmax) < 1e-20 * abs(vmax):
                    cs = ax.pcolormesh(
                        self.x.value, self.y.value, v.T, cmap=cmap, *args, **kwargs
                    )
                else:
                    cs = ax.contourf(
                        self.x.value,
                        self.y.value,
                        v.T,
                        levels=levels,
                        cmap=cmap,
                        extend="both",
                        *args,
                        **kwargs,
                    )
            if addgrid:
                if self.gencoord:
                    gx = self.x.value
                    gy = self.y.value
                else:
                    gg = np.meshgrid(self.x.value, self.y.value)
                    gx = np.reshape(gg[0], -1)
                    gy = np.reshape(gg[1], -1)

                ax.plot(gx, gy, "x")

            if showcolorbar:
                cb = f.colorbar(cs, ax=ax, pad=0.01)
                cb.formatter.set_powerlimits((0, 0))

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            title = varNames[isub]
            if varUnit != "dimensionless":
                title += " [" + varUnit + "]"
            if logscale:
                title = "$log_{10}$(" + title + ")"
            ax.set_title(title)

        if self.cut_norm != None and self.cut_loc != None:
            print("Plots at " + self.cut_norm + " = ", self.cut_loc)

        self.add_bottom_line(f, bottomline)

        return f, axes

    def add_contour(self, ax, var, unit="planet", rmask=None, *args, **kwargs):
        r"""Adding contour lines to an axis.

        Args:
            ax (matplotlib axis): the axis to plot contour lines.

            var (str): variable of contours.

        Examples:
            >>> f, axes = dc.plot("Bx<(50)>(-50) By (np.log(2*{rhos0}))>(-5)", xlim=[-40,-5])
            >>> dc.add_contour(axes[0,0], "rhos1>1")
        """

        vname, vmin, vmax = self.analyze_variable_string(var)

        ytVar = self.evaluate_expression(vname, unit)
        v = ytVar
        if type(ytVar) == yt.units.yt_array.YTArray:
            v = ytVar.value

        vmin = v.min() if vmin == None else vmin
        vmax = v.max() if vmin == None else vmax
        v = np.clip(v, vmin, vmax)

        if self.gencoord:
            triang = tri.Triangulation(self.x, self.y)
            if rmask != None:
                r = np.sqrt(self.x**2 + self.y**2)
                isbad = np.less(r, 1.2)
                mask = np.all(np.where(isbad[triang.triangles], True, False), axis=1)
                triang.set_mask(mask)
            ax.tricontour(triang, v.T, *args, **kwargs)
        else:
            ax.contour(self.x, self.y, v.T, *args, **kwargs)

    def add_stream(
        self,
        ax,
        var1,
        var2,
        density=1,
        nx=400,
        ny=400,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        rmask=None,
        *args,
        **kwargs,
    ):
        r"""
        Adding streamlines to an axis.

        Args:
            ax (matplotlib axis): the axis to add streamlines.

            var1/var2 (str): streamline variable names.

            density (float): it controls the number of streamlines.

        Examples:
            >>> f, axes = dc.plot("Bx<(50)>(-50) By (np.log(2*{rhos0}))>(-5)", xlim=[-40,-5])
            >>> dc.add_stream(axes[1,0], "Bx", "Bz", density=2)
        """

        v1 = self.evaluate_expression(var1).value
        v2 = self.evaluate_expression(var2).value
        if type(v1) == yt.units.yt_array.YTArray:
            v1 = v1.value
        if type(v2) == yt.units.yt_array.YTArray:
            v2 = v2.value

        if self.gencoord:
            if xmin == None:
                xmin = self.x.value.min()
            if xmax == None:
                xmax = self.x.value.max()
            if ymin == None:
                ymin = self.y.value.min()
            if ymax == None:
                ymax = self.y.value.max()

            gridx, gridy = np.mgrid[0 : nx + 1, 0 : ny + 1]
            gridx = gridx * (xmax - xmin) / nx + xmin
            gridy = gridy * (ymax - ymin) / ny + ymin
            xy = np.zeros((len(self.x), 2))
            xy[:, 0] = self.x.value
            xy[:, 1] = self.y.value
            # Remove the first and last row/column since they may be None
            vect1 = griddata(xy, v1, (gridx, gridy), method="linear")[1:-1, 1:-1]
            vect2 = griddata(xy, v2, (gridx, gridy), method="linear")[1:-1, 1:-1]
            xx = gridx[1:-1, 0]
            yy = gridy[0, 1:-1]
        else:
            xx = self.x.value
            yy = self.y.value
            vect1, vect2 = v1, v2

        if rmask != None:
            mask = np.zeros(vect1.shape, dtype=bool)
            r2 = rmask**2
            for i in range(len(xx)):
                for j in range(len(yy)):
                    if xx[i] ** 2 + yy[j] ** 2 < r2:
                        # vect1 and vect2 have been transposed
                        vect1[i, j] = np.nan
                        vect2[i, j] = np.nan

        streamplot(ax, xx, yy, vect1.T, vect2.T, density=density, *args, **kwargs)


class DataContainer1D(DataContainer):
    r"""
    A class handles 1D Cartesian data.
    """

    def __init__(self, dataSets, x, xlabel, *args, **kwargs):
        r"""
        Args:
            dataSets (dict): the keys are variable names, and the values are YTArrays.

            x: 1D YTArray

            xlabel (str): x axis label.
        """

        ylabel = None
        y = (0, 0)

        zlabel = None
        z = (0, 0)

        super(DataContainer1D, self).__init__(
            dataSets, x, y, z, xlabel, ylabel, zlabel, *args, **kwargs
        )

    def plot(
        self,
        vars,
        xlim=None,
        ylim=None,
        unit: str = "planet",
        figsize=(12, 8),
        bottomline=10,
        *args,
        **kwargs,
    ):
        """
        Examples:
            >>> f, axes = dc.plot("absdivb bx", xlim=[-5,5])
        """

        if type(vars) == str:
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

        f, axes = plt.subplots(nvar, 1, figsize=figsize)
        axes = np.array(axes)  # in case nrows == ncols == 1
        axes = axes.reshape(-1)

        for isub, ax in zip(range(nvar), axes):
            ytVar = self.evaluate_expression(varNames[isub], unit)
            v = ytVar

            vmin = v.min() if varMin[isub] == None else varMin[isub]
            vmax = v.max() if varMax[isub] == None else varMax[isub]
            # v = np.clip(v, vmin, vmax)
            ax.plot(self.x.value, v, label=varNames[isub], *args, **kwargs)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(self.xlabel)
            ax.legend()

        self.add_bottom_line(f, bottomline)

        return f, axes
