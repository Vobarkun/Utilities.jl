module Utilities

using Makie, IJulia


import REPL
REPL.REPLCompletions.latex_symbols["\\fig"] = "fig = Figure(); ax = Axis(fig[1,1])";
REPL.REPLCompletions.latex_symbols["\\figls"] = "fig = Figure(); ls = LScene(fig[1,1]; fixcam..., show_axis = false)";
REPL.REPLCompletions.latex_symbols["\\ls"] = "fig = Figure(); ls = LScene(fig[1,1]; fixcam..., show_axis = false)";
REPL.REPLCompletions.latex_symbols["\\angstrom"] = "Å";

function easein(x)
    x < 0 && return zero(x)
    x > 1 && return x - 0.5
    return x^3-x^4/2
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
    backend = Makie.current_backend()
    scene = Makie.get_scene(figlike)
    Makie.update_state_before_display!(figlike)
    screen = Makie.getscreen(backend, scene)
    display(screen, scene)
    f && focus()
end

function theme()
    Theme(
        resolution = (1000,1000), 
        Volume = (
            algorithm = :iso, 
            isovalue = 1.0, 
            isorange = 0.9, 
            colorrange = (0, 2)
        )
    )
end
set_theme!() = Makie.set_theme!(theme())


struct IntervalTicks step end
MakieLayout.get_tickvalues(t::IntervalTicks, vmin, vmax) = ceil(Int, vmin / t.step) * t.step : t.step : floor(Int, vmax / t.step) * t.step
const xlog10 = (xscale = log10, xticks = LogTicks(IntervalTicks(1)), xminorticksvisible=true, xminorticks = IntervalsBetween(9))
const ylog10 = (yscale = log10, yticks = LogTicks(IntervalTicks(1)), yminorticksvisible=true, yminorticks = IntervalsBetween(9))

xinc!(ax, xs...) = vlines!(ax, collect(xs), color = :transparent)
yinc!(ax, ys...) = hlines!(ax, collect(ys), color = :transparent)
include!(ax, ys) = scatter!(ax, xs, ys, color = :transparent)


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
        colormap = Reverse(:Set1_3),
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

        ts = isa(cs, AbstractArray) ? cs : range(0, 1, length = length(yss))

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
        lines!(sc, points, color = colors, colormap = sc.colormap, linewidth = sc.linewidth)
    else
        lines!(sc, points, color = sc.color, colormap = sc.colormap, linewidth = sc.linewidth)
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
        scene.camera = copy(scenes[1].camera)
        on(scene.camera_controls.eyeposition) do ep
            for scene2 in scenes
                scene2.camera_controls.eyeposition.val = ep
            end
        end
        on(scene.camera_controls.lookat) do ep
            for scene2 in scenes
                scene2.camera_controls.lookat.val = ep
            end
        end
        on(scene.camera_controls.upvector) do ep
            for scene2 in scenes
                scene2.camera_controls.upvector.val = ep
            end
        end
        on(scene.camera_controls.fov) do ep
            for scene2 in scenes
                scene2.camera_controls.fov.val = ep
            end
        end
        on(scene.camera_controls.zoom_mult) do ep
            for scene2 in scenes
                scene2.camera_controls.zoom_mult.val = ep
            end
        end
        on(scene.lights[1].position) do ep
            for scene2 in scenes
                scene2.lights[1].position.val = ep
            end
        end
    end
end

export window, IntervalTicks, xlog10, ylog10, xinc!, yinc!, include!, liftevery, linkCameras!, focus, easein, numpath, smoothstep

end # module Utils
