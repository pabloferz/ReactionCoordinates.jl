using ReactionCoordinates
using Documenter

makedocs(;
    modules=[ReactionCoordinates],
    authors="Pablo Zubieta <8410335+pabloferz@users.noreply.github.com> and contributors",
    repo="https://github.com/pabloferz/ReactionCoordinates.jl/blob/{commit}{path}#L{line}",
    sitename="ReactionCoordinates.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
