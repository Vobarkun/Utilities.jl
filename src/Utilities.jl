module Utilities

using Makie, IJulia, JLD2
import Makie.SpecApi as S

import REPL

function addREPLCompletions()
    REPL.REPLCompletions.latex_symbols["\\fig"] = "fig = Figure(); ax = Axis(fig[1,1])";
    REPL.REPLCompletions.latex_symbols["\\figls"] = "fig = Figure(); ax = LScene(fig[1,1]; fixcam..., show_axis = false)";
    REPL.REPLCompletions.latex_symbols["\\ls"] = "fig = Figure(); ax = LScene(fig[1,1]; fixcam..., show_axis = false)";
    REPL.REPLCompletions.latex_symbols["\\angstrom"] = "Å";
end

function easein(x)
    x < 0 && return zero(x)
    x > 1 && return x - 0.5
    return x * x * x * (1 - x / 2)
end

smoothstep(x) = ifelse(x < 0, zero(x), ifelse(x > 1, one(x), x * x * (3 - 2x)))
smoothstep(x, x0, x1) = smoothstep((x - x0) / (x1 - x0))
smoothstep(x, x0, x1, y0, y1) = y0 + (y1 - y0) * smoothstep(x, x0, x1)

function numpath(path)
    name, ext = splitext(path)
    i = 1
    while true
        !ispath("$name$i$ext") && return "$name$i$ext"
        i += 1
    end
end

function liftevery(innode, dt)
    t = Observable(time_ns())
    outnode = Observable(innode[])
    on(innode) do val
        if time_ns() - t[] > dt * 1e9
            t[] = time_ns()
            outnode[] = val
        end
    end
    outnode
end

function focus()
    display("text/javascript", "window.focus()")
    IJulia.clear_output()
    nothing
end

function window(figlike, f = true)
    # backend = Makie.current_backend()
    # scene = Makie.get_scene(figlike)
    # Makie.update_state_before_display!(figlike)
    # screen = Makie.getscreen(backend, scene)
    display(figlike)
    f && focus()
end

function wong_colors()
    Makie.wong_colors() .* 1.5
end

function wong_colors(i)
    wong_colors()[i]
end

function theme_dark()
    spinecolor = 1.1parse(Makie.RGB, "#4f5561")
    Theme(
        size = (1000,1000), 
        backgroundcolor = :transparent,
        linecolor = "#a8afbc",
        textcolor = "#a8afbc",
        Hist = (
            strokecolor = "#a8afbc",
            strokewidth = 1,
        ),
        BarPlot = (
            strokecolor = "#a8afbc",
            strokewidth = 1,
        ),
        palette = (color = wong_colors(),),
        Axis = (
            backgroundcolor = :transparent,
            bottomspinecolor = spinecolor,
            leftspinecolor = spinecolor,
            rightspinecolor = spinecolor,
            topspinecolor = spinecolor,
            xgridcolor = "#383e49",
            xminorgridcolor = ("#383e49", 0.5),
            xlabelpadding = 3,
            xminorticksvisible = false,
            xminortickcolor = spinecolor,
            xticksvisible = false,
            xtickcolor = spinecolor,
            ytickcolor = spinecolor,
            ygridcolor = "#383e49",
            yminorgridcolor = ("#383e49", 0.5),
            ylabelpadding = 3,
            yminorticksvisible = false,
            yminortickcolor = spinecolor,
            yticksvisible = false,
        ),
        Axis3 = (
            xgridcolor = (:white, 0.09),
            xspinesvisible = false,
            xticksvisible = false,
            ygridcolor = (:white, 0.09),
            yspinesvisible = false,
            yticksvisible = false,
            zgridcolor = (:white, 0.09),
            zspinesvisible = false,
            zticksvisible = false,
            backgroundcolor = :gray10,
        ),
        Colorbar = (
            spinewidth = 0,
            ticklabelpad = 5,
            ticksvisible = false,
        ),
        Legend = (
            framevisible = false,
            padding = (0, 0, 0, 0),
        ),
        Volume = (
            algorithm = :iso, 
            isovalue = 1.0, 
            isorange = 0.9, 
            colorrange = (0, 2)
        ),
        Mesh = (
            cycle = [:color => :color],
        ),
    )
end

function theme()
    Theme(
        size = (1000,1000), 
        Volume = (
            algorithm = :iso, 
            isovalue = 1.0,
            isorange = 0.9,
            colorrange = (0, 2)
        ),
        Mesh = (
            cycle = [:color => :color],
        ),
        Scatter = (fxaa = true, )
    )
end
set_theme!(t = :light; kwargs...) = Makie.set_theme!(t == :dark ? theme_dark() : theme(); kwargs...)

cam3dfixed!(scene; kwargs...) = cam3d!(scene; zoom_shift_lookat = false, kwargs...)
const fixcam = (scenekw = (camera = cam3dfixed!,),)

struct IntervalTicks step end
Makie.get_tickvalues(t::IntervalTicks, vmin, vmax) = (floor(Int, vmin / t.step) - 1) * t.step : t.step : (ceil(Int, vmax / t.step) + 1) * t.step

xinc!(ax, xs...) = vlines!(ax, collect(xs), color = :transparent)
yinc!(ax, ys...) = hlines!(ax, collect(ys), color = :transparent)
include!(ax, xs, ys) = scatter!(ax, xs, ys, color = :transparent)
include!(ax, ps) = scatter!(ax, ps, color = :transparent)

function ensurelogticks!(ax)
    (xmin, xmax), (ymin, ymax) = map((1, 2)) do i
        mapreduce(((x1, y1), (x2, y2)) -> (min(x1, x2), max(y1, y2)), ax.scene.plots) do plt
            extrema(getindex.(plt[1][], i))
        end
    end
    xmin -= 0.001xmin
    ymin -= 0.001ymin
    if ax.xscale[] == log10 && floor(log10(xmax)) - ceil(log10(xmin)) <= 0
        xinc!(ax, exp10(floor(Int, log10(xmin))), exp10(ceil(Int, log10(xmax))))
    end
    if ax.yscale[] == log10 && floor(log10(ymax)) - ceil(log10(ymin)) <= 0
        yinc!(ax, exp10(floor(Int, log10(ymin))), exp10(ceil(Int, log10(ymax))))
    end
end

function subfigure(fig, i, j; font = :bold, label = :automatic, xoffset = 20, yoffset = 0, kwargs...)
    sf = GridLayout(fig[i,j]);
    l = label == :automatic ? string.('a':'z')[length(contents(fig[:,:]))] : label
    Label(fig[i,j, TopLeft()], l;
        tellwidth = false, halign = :left, 
        padding = (xoffset, -xoffset, yoffset, -yoffset),
        font, kwargs...
    )
    sf
end

function linkedAxisGrid(func, figlike, nx, ny; kwargs...)
    axes = broadcast((1:nx)', 1:ny) do i, j
        ax = Axis(figlike[i,j]; kwargs...,
            xticksvisible = (i == nx), xticklabelsvisible = (i == nx), xlabelvisible = (i == nx),
            yticksvisible = (j == 1), yticklabelsvisible = (j == 1), ylabelvisible = (j == 1),
        )
        func(ax, i, j)
        # figlike[i,j] = ax
        ax
    end
    linkaxes!(axes...)
    axes
end

linkedAxisGrid(figlike, nx, ny; kwargs...) = linkedAxisGrid((ax, i, j) -> nothing, figlike, nx, ny; kwargs...)

function scientific(digits = 1)
    function format(ticks)
        map(ticks) do t
            if t == 0
                rich("0")
            else
                exponent = floor(Int, log10(abs(t)))
                significand = digits == 0 ? round(Int, t / exp10(exponent)) : round(t / exp10(exponent), digits = digits)
                rich((t < 0 ? "-" : ""), string(abs(significand)), " ⋅ 10", superscript(string(exponent)))
            end
        end
    end
end

struct PseudologTicks ticks end
Makie.get_tickvalues(t::PseudologTicks, vmin, vmax) = Makie.pseudolog10.inverse.(Makie.get_tickvalues(t.ticks, Makie.pseudolog10(10vmin), Makie.pseudolog10(10vmax)))

struct Pseudolog10Ticks end

function Makie.get_tickvalues(::Pseudolog10Ticks, vmin, vmax)
    ticks = Makie.pseudolog10.inverse.(ceil(Int, Makie.pseudolog10(10vmin)):1:Makie.pseudolog10(10vmax))
    ticks .= (ticks .+ sign.(ticks)) ./ 10
end

function Makie.get_ticklabels(::Pseudolog10Ticks, ticks)
    map(ticks) do t
        if t == 0
            rich("0")
        # elseif abs(t) == 1
            # rich(string(round(Int, t)))
        else
            rich((t < 0 ? "-" : "") * "10", superscript(string(round(Int, log10(abs(t))))))
        end
    end
end

function Makie.get_ticks(t::Pseudolog10Ticks, scale, tickformat, vmin, vmax)
    ticks = Makie.get_tickvalues(t, vmin, vmax)
    ticklabels = Makie.get_ticklabels(t, ticks)
    ticks, ticklabels    
end

const xlog10 = (xscale = log10, xticks = LogTicks(IntervalTicks(1)), xminorticksvisible=true, xminorticks = IntervalsBetween(9))
const ylog10 = (yscale = log10, yticks = LogTicks(IntervalTicks(1)), yminorticksvisible=true, yminorticks = IntervalsBetween(9))
const xasinh = (xscale = Makie.AsinhScale(1), xticks = Pseudolog10Ticks(), xminorticksvisible = true, xminorticks = IntervalsBetween(9))
const yasinh = (yscale = Makie.AsinhScale(1), yticks = Pseudolog10Ticks(), yminorticksvisible = true, yminorticks = IntervalsBetween(9))

function calculate_rgba(rgb1, rgb2, rgba_bg)::RGBAf
    rgb1 == rgb2 && return RGBAf(rgb1.r, rgb1.g, rgb1.b, 1)
    c1 = Float64.((rgb1.r, rgb1.g, rgb1.b))
    c2 = Float64.((rgb2.r, rgb2.g, rgb2.b))
    alphas_fg = 1 .+ c1 .- c2
    alpha_fg = clamp(sum(alphas_fg) / 3, 0, 1)
    alpha_fg == 0 && return rgba_bg
    rgb_fg = clamp.((c1 ./ alpha_fg), 0, 1)
    rgb_bg = Float64.((rgba_bg.r, rgba_bg.g, rgba_bg.b))
    alpha_final = alpha_fg + (1 - alpha_fg) * rgba_bg.alpha
    rgb_final = @. 1 / alpha_final * (alpha_fg * rgb_fg + (1 - alpha_fg) * rgba_bg.alpha * rgb_bg)
    return RGBAf(rgb_final..., alpha_final)
end

function alpha_colorbuffer(figure)
    scene = figure.scene
    bg = scene.backgroundcolor[]
    scene.backgroundcolor[] = RGBAf(0, 0, 0, 1)
    b1 = copy(colorbuffer(scene))
    scene.backgroundcolor[] = RGBAf(1, 1, 1, 1)
    b2 = colorbuffer(scene)
    scene.backgroundcolor[] = bg
    return map(b1, b2) do b1, b2
        calculate_rgba(b1, b2, RGBAf(1,1,1,0))
    end
end



@recipe(MultiLines) do scene
    Attributes(
        colormap = cgrad(Makie.wong_colors()[1:2]),
        colorrange = Makie.Automatic(),
        linewidth = 2,
        color = nothing
    )
end



function Makie.plot!(sc::MultiLines{<:Tuple{AbstractVector{<:Real}, AbstractVector{<:AbstractVector{<:Real}}}})
    xs = sc[1]
    yss = sc[2]
    cs = sc.color

    points = Observable(Point2f[])
    colors = Observable(Float64[])

    function update_plot(xs, yss, cs)
        empty!(points[])
        empty!(colors[])

        ts = isa(cs, AbstractArray) ? cs : range(0, ifelse(length(yss) > 1, 1, 0), length = length(yss))

        for (t, ys) in zip(ts, yss)
            append!(points[], Point2f.(xs, ys))
            push!(points[], Point2f(NaN, NaN))
            append!(colors[], fill(t, length(ys) + 1))
        end

        colors[] = colors[]
        points[] = points[]
    end

    Makie.Observables.onany(update_plot, xs, yss, cs)

    update_plot(xs[], yss[], cs[])
    if isnothing(cs[]) || isa(cs[], AbstractArray)
        lines!(sc, points, color = colors, colormap = sc.colormap, colorrange = sc.colorrange, linewidth = sc.linewidth)
    else
        lines!(sc, points, color = sc.color, colormap = sc.colormap, colorrange = sc.colorrange, linewidth = sc.linewidth)
    end

    sc
end

function Makie.convert_arguments(P::Type{<:MultiLines}, yss::AbstractVector{<:AbstractVector{<:Real}})
    (1:length(yss[1]), yss)
end

function Makie.convert_arguments(P::Type{<:MultiLines}, xs, yss::AbstractMatrix{<:Real})
    (xs, collect(eachcol(yss)))
end

function Makie.convert_arguments(P::Type{<:MultiLines}, yss::AbstractMatrix{<:Real})
    (1:size(yss, 1), collect(eachcol(yss)))
end

# Makie.preferred_axis_type(::MultiVolume) = LScene
# @recipe(MultiVolume) do scene
#     Theme()
# end

# function Makie.plot!(plt::MultiVolume)
#     println(typeof(plt))
#     # volume!(plt, plt[1][1])
#     # plot!(plt, S.LScene(plots = [S.Volume(plt[1])]))
#     plt
# end

# function Makie.convert_arguments(P::Type{<:MultiVolume}, vols::Array)
#     g = S.GridLayout([S.LScene(plots = [S.Volume(v)]) for v in vols])
#     println(typeof(g))
#     (g, vols)
# end


# plot!(plt::MultiVolume) = plt


linkCameras!(g::GridPosition) = linkCameras!(contents(g))
linkCameras!(scenes...) = linkCameras!(scenes)
function linkCameras!(contents)
    scenes = map(contents) do s
        if s isa LScene
            s.scene
        else
            s
        end
    end
    linkCams!(scenes)
end

function linkCams!(scenes)
    for scene in scenes[1:end]
        on(scene.camera_controls.eyeposition) do ep
            for scene2 in scenes
                scene2.camera_controls.eyeposition.val = ep
                update_cam!(scene2)
            end
        end
        on(scene.camera_controls.lookat) do ep
            for scene2 in scenes
                scene2.camera_controls.lookat.val = ep
                update_cam!(scene2)
            end
        end
        on(scene.camera_controls.upvector) do ep
            for scene2 in scenes
                scene2.camera_controls.upvector.val = ep
                update_cam!(scene2)
            end
        end
        on(scene.camera_controls.fov) do ep
            for scene2 in scenes
                scene2.camera_controls.fov.val = ep
                update_cam!(scene2)
            end
        end
    end
end

function mapflat(f, arrs)
    result = Vector{eltype(f.(arrs[1]))}(undef, sum(length, arrs))
    k = 1
    for arr in arrs, a in arr
        result[k] = f(a)
        k += 1
    end
    result
end

function twinx(ax; tickformat = k -> (k != 0) ? string(round(2pi / k, digits = 1)) : "∞", kwargs...)
    gc = ax.layoutobservables.gridcontent[]
    ax2 = Axis(gc.parent[gc.span.rows, gc.span.cols];
        xaxisposition = :top, xticks = ax.xticks,
        xtickformat = ks -> tickformat.(ks),
        kwargs...
    )
    hideydecorations!(ax2)
    linkaxes!(ax, ax2)
    ax2
end

iscanceled(fig) = ispressed(fig, Keyboard.escape)

function posFig(ax, points::AbstractArray{<:AbstractArray})
    lift(ax.scene.viewport, ax.layoutobservables.computedbbox, ax.scene.camera.projectionview) do px_area, bbox, _
		Makie.project.(ax.scene, Point2f.(points)) .+ (px_area.origin,)
	end
end
function posFig(ax, x, y)
    lift(ax.scene.viewport, ax.layoutobservables.computedbbox, ax.scene.camera.projectionview) do px_area, bbox, _
		Makie.project(ax.scene, Point2f(x, y)) + px_area.origin
	end
end
posFig(ax, xy) = posFig(ax, xy...)

function collect_labeled_plots(figlike, unique = true, sortbytype = true)
    x = map(filter(x -> isa(x, Axis), contents(figlike))) do ax
        Makie.get_labeled_plots(ax, merge = false, unique = false)
    end
    plots_in_fig = reduce(vcat, first.(x))
    labels_in_fig = reduce(vcat, last.(x))
    if !unique
        return plots_in_fig, labels_in_fig
    end
    ulabels = Base.unique(labels_in_fig)
    uplots = [plots_in_fig[findfirst(isequal(l), labels_in_fig)] for l in ulabels]

    if sortbytype
        perm = sortperm(uplots, by = string ∘ typeof)
        ulabels = ulabels[perm]; uplots = uplots[perm]
    end

    uplots, ulabels
end

function cmap(name)
    if name == :wong12
        return cgrad(Makie.wong_colors()[1:2])
    elseif name == :darkisoluminant
        return cgrad([Makie.HSV(x.h, 1.5x.s, 0.9x.v) for x in Makie.HSV.(cgrad(:isoluminant_cm_70_c39_n256))])
    end
    jldopen(normpath(joinpath(@__DIR__, "..", "data/cmaps.jld2"))) do f
        cgrad(f[string(name)])
    end
end

function padmaxlength(s)
    maxl = maximum(length, s)
    rpad.(s, maxl)
end

function fieldinfo(s)
    fields = fieldnames(typeof(s))
    types = typeof.(getfield.((s,), fields))
    sizes = map(fields) do f
        try 
            return size(getfield(s, f))
        catch e
            return ""
        end
    end
    
    fieldstrings = padmaxlength(string.(fields))
    typestrings = padmaxlength(string.(types))
    typestrings = [t[1:min(50, end)] for t in typestrings]
    sizestrings = padmaxlength(string.(sizes))

    for t in zip(fieldstrings, typestrings, sizestrings)
        for x in t
            print(x, "  ")
        end
        println()
    end
end

macro runtostr(expr)
    s = join(filter(x -> !occursin("#=", x), split(string(expr), "\n")), "\n")
    return esc(quote
        $expr
        $s
    end)
end

export window, IntervalTicks, xlog10, ylog10, xinc!, yinc!, include!, liftevery, linkCameras!, focus, easein, numpath, smoothstep, fixcam, cam3dfixed!, addREPLCompletions, mapflat, twinx, iscanceled, Pseudolog10Ticks, linkedAxisGrid, subfigure, scientific, ensurelogticks!, fieldinfo, @runtostr, posFig

end # module Utils
