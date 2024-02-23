push!(LOAD_PATH, "../src/")

using Documenter, OpiForm

makedocs(
  sitename="OpiForm Documentation",
  repo=Remotes.GitHub("gjankowiak", "OpiForm_alpha")
)
