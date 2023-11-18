module Utils

using Makie, IJulia

function easein(x)
    x < 0 && return zero(x)
    x > 1 && return x - 0.5
    return x^3-x^4/2
end

function smoothstep(x)
    x = clamp(x, zero(x), one(x))
    x * x * (3 - 2x)
end

function numpath(path)
    name, ext = splitext(path)
    i = 1
    while true
        !ispath("$name$i$ext") && return "$name$i$ext"
        i += 1
    end
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

struct IntervalTicks
    step
end
MakieLayout.get_tickvalues(t::IntervalTicks, vmin, vmax) = ceil(Int, vmin / t.step) * t.step : t.step : floor(Int, vmax / t.step) * t.step

const xlog10 = (xscale = log10, xticks = LogTicks(IntervalTicks(1)), xminorticksvisible=true, xminorticks = IntervalsBetween(9))
const ylog10 = (yscale = log10, yticks = LogTicks(IntervalTicks(1)), yminorticksvisible=true, yminorticks = IntervalsBetween(9))

xinc!(ax, xs...) = vlines!(ax, collect(xs), color = :transparent)
yinc!(ax, ys...) = hlines!(ax, collect(ys), color = :transparent)
include!(ax, ys) = scatter!(ax, xs, ys, color = :transparent)

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

    ls = Observable(Point2f[])
    colors = Observable(Float64[])

    function update_plot(xs, yss, cs)
        colors[]

        empty!(ls[])
        empty!(colors[])

        cs = isnothing(cs) ? range(0, 1, length = length(yss)) : cs

        for (t, ys) in zip(cs, yss)
            append!(ls[], Point2f.(xs, ys))
            push!(ls[], Point2f(NaN, NaN))
            append!(colors[], fill(t, length(ys) + 1))
        end

        colors[] = colors[]
    end

    Makie.Observables.onany(update_plot, xs, yss, cs)

    update_plot(xs[], yss[], cs[])

    # if !isnothing(sc.color[])
    #     lines!(sc, ls, color = sc.color, colormap = sc.colormap, linewidth = sc.linewidth)
    # else
    lines!(sc, ls, color = colors, colormap = sc.colormap, linewidth = sc.linewidth)
    # end

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
