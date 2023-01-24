# A simulation of the growth of barolinic instability in the Phillips 2-layer model
# when we impose a vertical mean flow shear as a difference ``\Delta U`` in the
# imposed, domain-averaged, zonal flow at each layer.
using GeophysicalFlows, CairoMakie, Printf
using FourierFlows: parsevalsum
using Random: seed!
using CUDA

# ## Device
dev = GPU()
if dev == CPU()
  Random.seed!(1234)
else
  CUDA.seed!(1234)
end
random_uniform = dev == CPU() ? rand : CUDA.rand

# ## Parameters
n = 128                  # 2D resolution = n²
stepper = "FilteredRK4"  # timestepper
dt = 2.5e-3              # timestep
nsteps = 20000           # total number of time-steps
nsubs = 50               # number of time-steps for plotting (nsteps must be multiple of nsubs)
L = 2π                   # domain size
nν = 4
ν = 1e-14
μ = 1e-1                 # bottom drag
β = 0                    # the y-gradient of planetary PV
nlayers = 1              # number of layers
f₀, g = 0, 0             # Coriolis parameter and gravitational constant
H = [1.0]                # the rest depths of each layer
ρ = [1.0]                # the density of each layer
U = zeros(nlayers)       # the imposed mean zonal flow in each layer
U[1] = 0.0

# ## Stochastic forcing
forcing_wavenumber = 14.0 * 2π / L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth = 1.5 * 2π / L  # the width of the forcing spectrum, `δ_f`
ε = 0.1                           # energy input rate by the forcing
grid = TwoDGrid(dev; nx=n, Lx=L)
K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber
forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
CUDA.@allowscalar forcing_spectrum[grid.Krsq.==0] .= 0 # ensure forcing has zero domain-average
ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε / ε0        # normalize forcing to inject energy at rate ε

function calcFq!(Fh, sol, t, clock, vars, params, grid)
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol))) ./ sqrt(clock.dt)

  return nothing
end

# ## Deterministic Forcing

# ## Problem setup
prob = MultiLayerQG.Problem(nlayers, dev; nx=n, Lx=L, f₀, g, H, ρ, U, μ, β, ν=ν, nν=nν,
  dt, stepper, aliased_fraction=0, calcFq=calcFq!, stochastic=true)

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = grid.x, grid.y

# ## Setting initial conditions
q₀ = 0 * device_array(dev)(randn((grid.nx, grid.ny, nlayers)))
q₀h = prob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
q₀ = irfft(q₀h, grid.nx, (1, 2))                 # apply irfft only in dims=1, 2
MultiLayerQG.set_q!(prob, q₀)

# ## Diagnostics
E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.

# Helper function that extracts the Fourier-transformed solution
get_sol(prob) = prob.sol

function get_u(prob)
  sol, params, vars, grid = prob.sol, prob.params, prob.vars, prob.grid

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l * vars.ψh
  invtransform!(vars.u, vars.uh, params)

  return vars.u
end

# ## Visualizing the simulation
Lx, Ly = grid.Lx, grid.Ly

title_KE = Observable(@sprintf("μt = %.2f", μ * clock.t))

q₁ = Observable(Array(vars.q[:, :, 1]))
ψ₁ = Observable(Array(vars.ψ[:, :, 1]))

function compute_levels(maxf, nlevels=8)
  ## -max(|f|):...:max(|f|)
  levelsf = @lift collect(range(-$maxf, stop=$maxf, length=nlevels))

  ## only positive
  levelsf⁺ = @lift collect(range($maxf / (nlevels - 1), stop=$maxf, length=Int(nlevels / 2)))

  ## only negative
  levelsf⁻ = @lift collect(range(-$maxf, stop=-$maxf / (nlevels - 1), length=Int(nlevels / 2)))

  return levelsf, levelsf⁺, levelsf⁻
end

maxψ₁ = Observable(maximum(abs, vars.ψ[:, :, 1]))

levelsψ₁, levelsψ₁⁺, levelsψ₁⁻ = compute_levels(maxψ₁)

KE₁ = Observable(Point2f[(μ * E.t[1], E.data[1][1][1])])

fig = Figure(resolution=(1000, 600))
axis_kwargs = (xlabel="x",
  ylabel="y",
  aspect=1,
  limits=((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)))

axq₁ = Axis(fig[1, 1]; title="q₁", axis_kwargs...)

axψ₁ = Axis(fig[2, 1]; title="ψ₁", axis_kwargs...)

axKE = Axis(fig[1, 3],
  xlabel="μ t",
  ylabel="KE",
  title=title_KE,
  yscale=log10,
  limits=((-0.1, 2.6), (1e-9, 5)))

heatmap!(axq₁, x, y, q₁; colormap=:balance)

contourf!(axψ₁, x, y, ψ₁;
  levels=levelsψ₁, colormap=:viridis, extendlow=:auto, extendhigh=:auto)
contour!(axψ₁, x, y, ψ₁;
  levels=levelsψ₁⁺, color=:black)
contour!(axψ₁, x, y, ψ₁;
  levels=levelsψ₁⁻, color=:black, linestyle=:dash)

ke₁ = lines!(axKE, KE₁; linewidth=3)
Legend(fig[1, 4], [ke₁,], ["KE₁",])


# ## Time-stepping the `Problem` forward
startwalltime = time()
frames = 0:round(Int, nsteps / nsubs)

CairoMakie.record(fig, "multilayerqg_2layer.mp4", frames, framerate=18) do j
  # update terminal
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE₁: %.3e, PE: %.3e, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i][1][1], E.data[E.i][1][1], (time() - startwalltime) / 60)

    println(log)
  end

  # update observables
  q₁[] = vars.q[:, :, 1]
  ψ₁[] = vars.ψ[:, :, 1]
  maxψ₁[] = maximum(abs, vars.ψ[:, :, 1])
  KE₁[] = push!(KE₁[], Point2f(μ * E.t[E.i], E.data[E.i][1][1]))
  title_KE[] = @sprintf("μ t = %.2f", μ * clock.t)

  # step simulation forward
  stepforward!(prob, diags, nsubs)
  MultiLayerQG.updatevars!(prob)
end
