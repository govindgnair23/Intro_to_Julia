### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 7eaa9450-1e09-11ec-175c-41430c2f547f
using Images, TestImages, ImageFiltering, Statistics, PlutoUI, BenchmarkTools

# ╔═╡ 6bb7a060-effe-11eb-02ce-c704b0dccd0c
using OffsetArrays, Random, Plots, ImageTransformations

# ╔═╡ 165932eb-2992-4580-947d-33a861d9007e
PlutoUI.TableOfContents(title="📚 Table of Contents",aside=true)

# ╔═╡ 2341517a-93e8-4c8c-afcc-87537776c09c
md"""
## Why a new language ?
"""

# ╔═╡ 467f60dc-26c0-411b-a548-0547f4ec4217
md"""

This is what Julia's creators had to say . See [here](https://julialang.org/blog/2012/02/why-we-created-julia/) for blog post.

"""

# ╔═╡ 2bab0ae8-3283-48d0-a17c-7a1b812e71c8
md"""

> We want a language that's open source, with a liberal license. We want the speed of C with the dynamism of Ruby. We want a language that's homoiconic, with true macros like Lisp, but with obvious, familiar mathematical notation like Matlab. We want something as usable for general programming as Python, as easy for statistics as R, as natural for string processing as Perl, as powerful for linear algebra as Matlab, as good at gluing programs together as the shell. Something that is dirt simple to learn, yet keeps the most serious hackers happy. We want it interactive and we want it compiled.
>
>(Did we mention it should be as fast as C? )

"""

# ╔═╡ 6e0220d1-a566-4a2c-9a4b-4a5f8c55cdfb
md"""

## Why Julia ?

1) Julia is really fast!
"""

# ╔═╡ ff9a7a8f-160b-4438-a327-1cd672075725
begin
md"""
	
![](https://julialang.org/assets/benchmarks/benchmarks.svg)
	
"""
end

# ╔═╡ cafc8053-4ca7-4105-941b-6d8774366e21
md"""
Julia is one of the few languages that are in the so-called PetaFlop family; the other languages are C, C++ and Fortran. It achieved 1.54 petaflops with 1.3 million threads on the Cray XC40 supercomputer.Source: [link](https://www.avenga.com/magazine/julia-programming-language/)

"""

# ╔═╡ 17603d7f-d337-4b6b-94c1-d304c951dabc
md"""

2. It is Julia all the way down!

Listen more to Jeremy Howard explaining why Python may not be the future of ML
[Link](https://www.youtube.com/watch?v=t2V2kf2gNnI&t=2435s)
"""

# ╔═╡ 045b7478-32a1-4d9e-b1ab-2b24a2d97601
md"""

## Load essential packages
"""

# ╔═╡ 2d62de89-d56e-4483-abc4-53286fabff65
md"""
## Multiple Dispatch : A Key Paradigm in Julia
"""

# ╔═╡ fc593ffa-6bc1-4ac9-8d63-3475ddf85067
md"""
See this [video](https://www.youtube.com/watch?v=kc9HwsxE1OY&t=1601s) for more details.
"""

# ╔═╡ 6fec836e-b95f-4fb7-950f-0f8703662e0b
md"""

### Simple Example

A simple example of how multiple dispatch works.
"""

# ╔═╡ b44a8d8d-fbef-4a17-9154-4b9bc76c2e05
begin
	abstract type Pet end
	
	struct Dog <: Pet
		name::String
	end
	
	struct Cat <: Pet
		name::String
	end
	
	meets(a::Dog,b::Dog) = "sniffs"
	meets(a::Dog,b::Cat) = "chases"
	meets(a::Cat,b::Dog) = "hisses"
	meets(a::Cat,b::Cat) = "slinks"
	
	function encounter(a::Pet, b::Pet)
		verb = meets(a,b)
		with_terminal() do
			println("$(a.name) meets $(b.name) and $verb")
		end
		#return "$(a.name) meets $(b.name) and $verb"
		
	end
		

end

# ╔═╡ 8e9875cb-0186-44ff-92e8-709684b64dae
begin
	fido = Dog("Fido")
	rex = Dog("Rex")
	whiskers = Cat("Whiskers")
	spots = Cat("Spots")
end

# ╔═╡ 86e38c22-4fd8-4b8a-8fd3-468c971925f9
encounter(fido,rex)

# ╔═╡ 2596db79-9244-4015-bfc3-c9444ca0d9c5
encounter(fido,whiskers)

# ╔═╡ 77949b67-42ad-4567-9391-c2f384bc733a
encounter(whiskers,rex)

# ╔═╡ 52edb6b6-ad57-47df-8a79-68f4003c7972
encounter(whiskers,spots)

# ╔═╡ 5d31ad64-d7c1-42bb-a1ba-fc3add097ad7
md"""
Python does have en external library that supports multiple dispatch but is less powerful.

"""

# ╔═╡ f37a2cbf-690d-4161-8dd3-c250760d8fa8
md"""

### Realistic Example

"""

# ╔═╡ 64e3e5a3-be0f-4c8f-83f3-ceb093cd2f03
md"""
Let us assume we are implementing matrix vector multiplication. A simple (possibly non-optimal) way to implement matrix vector multiplication is shown below.

"""

# ╔═╡ 2698e5c2-f3c0-40f3-9cd9-239e600a17ec
begin
	mm = download("https://hadrienj.github.io/assets/images/2.2/dot-product.png")
	load(mm)
end

# ╔═╡ 38177084-bbb1-4f75-886c-89aa6686ed47
function mult(M::AbstractMatrix,v::AbstractVector)
	@assert size(M,2) == length(v)
	return(reduce(+,[M[:,i]*v[i] for i in range(1,stop=length(v))]))
end

# ╔═╡ 518820ba-9254-4750-9675-eb5b369b6908
md"""
Now let us say we have a one hot vector . Clearly the above approach is inefficent when the vector to be multipied is a one-hot vector
"""

# ╔═╡ 12de93bb-1811-44f8-af21-3595360a0b0d
md"""
We can create a new one-hot vector type that is a sub type of **AbstractVector** and hence behaves like a vector
"""

# ╔═╡ dd22d4ef-822e-4272-a71d-650ff47eb32a
struct OneHot <: AbstractVector{Int}
	n::Int #Length of vector
	k::Int # Element correspinding to 1
end

# ╔═╡ 13c210b7-6d90-4a3a-a088-ecbd069f3de5
md"""
By defining the below two methods, the one hot vector type can be used like a regular vector
"""

# ╔═╡ 40daf36b-2de3-4f27-978b-a57a6492d0c5
Base.size(x::OneHot) = (x.n,)

# ╔═╡ a5316ebc-9663-45a5-99b6-dfac03d2a1d5
Base.getindex(x::OneHot,i::Int) = Int(x.k==i)

# ╔═╡ 6d631da4-e32b-407b-85c4-a3dabe37e113
oh = OneHot(2,2)

# ╔═╡ fe365df6-f021-4a4f-9699-28d0672e169c
oh2 = OneHot(5,3)

# ╔═╡ 3b2ad486-5d37-4f03-840c-85ebcba3096b
md"""
We can now define a new matrix vector multiplication function that works more efficiently with one hot vectors
"""

# ╔═╡ 261acfba-26ca-4f62-82c3-8beea15785ab
function mult(M::AbstractMatrix,v::OneHot)
	@assert size(M,2) == length(v) #length now works like in a regular vector
	return M[:,v.k]
end

# ╔═╡ 5f55eb45-da6c-4ecc-941f-68ba9fcbd613
begin
	M = [1 2
	     3 4
	     5 6]
	v = [1;2]
	mult(M,v)
end

# ╔═╡ bd1f8933-0791-43dd-a03d-83ba4a2cf6be
M*v

# ╔═╡ c8a7adb0-6ea4-41bd-84b3-5dcb1f02dd7f
mult(M,oh)

# ╔═╡ 5f73bd5d-f247-44e9-a1f6-3e2538e44717
md"""
## Useful Julia Tips

"""

# ╔═╡ ee3f88f8-8a06-43fc-9fb7-a337d298c6b8
md"""

### Arrays
"""

# ╔═╡ 42244456-dd4a-4f27-8d78-fd1a07dfaa29
numbers = collect(1:5)

# ╔═╡ f424daaa-753c-46dc-96ea-3dc7dc6c600f
numbersandletters = [1,"a"]

# ╔═╡ 35a2e4a0-d3a9-4d71-a96a-79983239f4de
md"""
Broadcasting works using the . operator
"""

# ╔═╡ 10f9b836-3d37-490b-8e10-02a0995e9e51
numbers .+ 2

# ╔═╡ 33c5b3ec-912e-4f9d-9464-737b4330c283
colors = [RGB(1.0,0,0),RGB(0,0,1),RGB(1,0,1),RGB(0.75,0,0.6)]

# ╔═╡ dfd32582-25da-4542-bcaa-1c1e0c508a75
md"""
Note indexing starts at 1 
"""

# ╔═╡ 5e4da1c5-7cf0-40b7-a383-1caabe45e1ff
numbers[1:end]

# ╔═╡ 79e08671-7bad-4aa4-898b-a2e5c72f886f
md"""
Array comprehensions are identical to list/dictionary comprehenions in Python
"""

# ╔═╡ 57cb5c8d-52e2-4af7-9519-00feec2aade6
[RGB(color.r,0.5,color.b) for color in colors]

# ╔═╡ 6c2b94cb-0ed3-4316-a3a7-05aab39ba9c9
reshape(colors,(2,2))

# ╔═╡ 0ca86629-9aea-482a-af01-f159d06ee5a6
md"""

### Functions
"""

# ╔═╡ fc7c29ce-67bb-4814-ac46-3fd9c87ba7bd
md"""

#### Short Form

"""

# ╔═╡ 4f1036b4-ad87-4fa5-aa93-7ea5f29aa56d
begin
	area(r) = π*r^2
	area(1)
end

# ╔═╡ ad2a5d50-7757-4e67-8335-a9cfa8677943
md"""
#### Anonymous Form
"""

# ╔═╡ 6ac64b90-b7a8-4db0-8976-8bb79ecf915f
(r -> π*r^2)(1)

# ╔═╡ f60287e0-fe3d-4c5c-8084-dcc258afc41a
function area(r::Int64)
	return π*r^2
end

# ╔═╡ e1e0e955-ec30-4f11-9f59-7025ecf3c86c
area(1)

# ╔═╡ 1439a19f-1f85-4407-9790-7732c35bde64
md"""

### Dictionaries
"""

# ╔═╡ 89c51fe2-1023-45e9-a0b5-dba8b165b3f7
md""" 
Dictionaries are a collection of pairs
"""

# ╔═╡ 66fb8c0a-b050-4a75-9f55-151eaca0c836
typeof( "a" => 1)

# ╔═╡ debd6aed-c218-4ae5-a97d-7afdfe9943ca
D = Dict("a"=> 1, "b"=> 2, "c"=> 3)

# ╔═╡ cdef8039-fa96-4f79-935f-8050c047c858
get(D,"x",0),D["a"]

# ╔═╡ c406e5b1-d81d-49e0-8401-4369eb2714c5
md"""
You can also create a list of pairs
"""

# ╔═╡ 74a2b652-e68d-4d03-ac66-4dd039bcf418
md"""
### Offset Arrays
"""

# ╔═╡ 4b6ab118-31ff-4d33-bbb1-532c2bbac8cc
md"""

Julia supports arbitrary indexing found in languages like Fortran.
"""

# ╔═╡ 8f64b69b-e2ac-4831-b874-6dd8aefac417
vec1 = OffsetVector(collect(1:5),0:4) 

# ╔═╡ 5e3a09a2-a64d-4ded-985c-3067423cae57
vec1[0]

# ╔═╡ 2bea3f87-dd6e-411a-8088-9e7e19c10da4
vec2 = OffsetVector(collect(1:5),-2:2) 

# ╔═╡ 3ed9bfc8-c59d-4aa2-a614-3844530e7b13
vec2[-2]

# ╔═╡ 3ca46cf8-6b70-4422-a264-3fe1b6658503
md"""
## Seam Carving
"""

# ╔═╡ 0fa9affd-b887-459d-a3a3-11ce0087ec19
md"""
[Seam carving](https://faculty.idc.ac.il/arik/SCWeb/imret/imret.pdf) is a content aware image re-sizing algorithm. It can resize images by preserving the sailent features of an image in its original size.
"""

# ╔═╡ 0d605a3e-f898-46a7-9932-085a48287cb4
md"""

The algorithm works as follows

1. Assign each pixel a value signifying its importance in the image
2. Find a seam - a connected path of pixels from top to bottom of the image - comprising the least important pixels
3. Remove the seam.


"""

# ╔═╡ 97be249a-3105-4ea6-81a3-2077ef2ad242
md"""
Consider "The Persistence of Memory" by Salvador Dali
"""

# ╔═╡ 714bfd1d-0a0b-4a62-bdfe-1d3be6d9e878
@bind ratio Slider(0.1:0.1:1,show_value=true)

# ╔═╡ 7e0f5420-d0dc-4b1d-8eb2-8962996eb483
begin
	img_url = "https://wisetoast.com/wp-content/uploads/2015/10/The-Persistence-of-Memory-salvador-deli-painting.jpg"
	image = load(download(img_url))
	image_small = imresize(image,ratio=ratio)
end

# ╔═╡ 8d463199-243f-4f29-95ea-45ffd1509bd0
md"""
### Kernels and Convolutions
"""

# ╔═╡ 03933a38-a1a1-4d37-acaf-2a49a4eaa334
md"""
We can assign an importance value to each pixel by [convolving](https://youtu.be/8rrHTtUzyZA) an edge detection filter such as the Sobel filter over the image.
"""

# ╔═╡ e2288fca-ff98-4702-b159-eb2605abe805
convolve(img, k) = imfilter(img, reflect(k)) 

# ╔═╡ 152cc6f4-684a-4458-a120-49158668be52
Ky,Kx = Kernel.sobel()

# ╔═╡ 3741074c-46e9-4455-80ae-f4d15978030c
md"""
Applying the first filter measures gradient along the y-axis the allows us to identify edges in the horizontal direction
"""

# ╔═╡ 1dd22345-89f7-4681-a1d8-236f3198eaba
begin
	∇y = convolve(image_small,Ky).*5
end

# ╔═╡ e390d90e-73de-4e37-a9e4-accf405f15a2
md"""
Applying the second filter allows us to measure the gradient in the x-direction
allowing us to identify edges in the vertical direction
"""

# ╔═╡ 70d7ea26-8701-4d41-b34b-4f8065092534
begin
	∇x = convolve(image_small,Kx).*5 
end

# ╔═╡ 376ad3f6-7f95-4167-b2be-fdc460419ee9
md"""

We can combine these two to give a sense of the overall importance of each pixel.However taking powers and square roots on RGB objects are not defined so we can define new methods to do just this.
"""

# ╔═╡ a5122b62-be88-494a-9cfe-09603b308b9e
begin
	Base.:^(x::RGB{Float64},y::Integer) = RGB(x.r^y,x.g^y,x.b^y)
	Base.sqrt(x::RGB{Float64}) = 	RGB(sqrt(max(0,x.r)),sqrt(max(0,x.g)),max(0,sqrt(x.b)))
	
end

# ╔═╡ 5cf9beed-6864-4c3b-863b-947fb8c1e14f
∇ = sqrt.((∇x).^2 .+ (∇y).^2)

# ╔═╡ d2fc48ec-8b9e-42a2-af9b-f3a7617377de
md"""
We still have to assign each pixel in the image a single scalar value that represents its importance in the image. A brightness function is defined to do this.
"""

# ╔═╡ 6c7e4b54-f318-11ea-2055-d9f9c0199341
begin
	brightness(c::RGB) = mean((c.r, c.g, c.b))
	brightness(c::RGBA) = mean((c.r, c.g, c.b))
	brightness(c::Gray) = gray(c)
end

# ╔═╡ 89adfdaa-376e-4d5b-a0d0-08d84b0dd810
Gray.(brightness.(∇))

# ╔═╡ ae1c488f-e4c6-4301-8998-e1b392869a56
md"""
We have now completed Step 1 of the algorithm. We have assigned each pixel a value,(an energy) signifying its importance to the image. We can now wrap these into a single function
"""

# ╔═╡ cb72bf7f-c98a-42a8-8c32-8b1bccdffb58
function energy2(img)
	∇x = convolve(image_small,Kx)
	∇y = convolve(image_small,Ky)
	∇ = sqrt.((∇x).^2 .+ (∇y).^2)
	return brightness.(∇)		
end

# ╔═╡ f404877d-ddce-4586-83f7-a0298a8c1c7b
md"""

### Finding Seams

"""

# ╔═╡ a74c03a2-eae7-4504-a63b-5395d794c983
md"""
The next step in the Seam Carving algorithm is to find a connected seam from top to bottom with the least important or the lowest energy pixels.
"""

# ╔═╡ eb20bc08-ed6e-4dde-a69a-4c29ff4d6363
md"""

A connected seam is defined by the following possible connections. You can connect only to 
1) The cell directly below
2) The cell to the left of 1.
3) The cell to the right of 1.

"""

# ╔═╡ cec83331-ed7f-4c5d-b5b9-f0600929e31b
md"""
Consider a matrix of numbers as shown below.
A path is a connected seam from top to bottom, the total energy of the seam is the sum of the numbers in the path.
"""

# ╔═╡ 157d45c9-a3e0-4d46-bb19-1aa8d4955f45
md"""
n =  $(@bind n Slider(2:12, show_value = true, default = 8))
"""

# ╔═╡ 05654405-1991-48f8-bb8c-76d61596deec
begin
	Random.seed!(99)
	M1 = rand(0:10,n,n)
end

# ╔═╡ 9a6a0b68-6eaa-4d92-9535-ffd5c9e6446f
md"""
We can define a Paths objects, iterating over the object will give all the possible seams in the matrix.
"""

# ╔═╡ 6b1ef6b2-2f0b-42e9-b87c-759e8b200543
begin
	struct Paths
		m::Int #Number of rows
		n::Int #Number of columns
	end
end

# ╔═╡ 8b19990f-e55e-4255-8921-d202c46f9569
begin
	Base.iterate(p::Paths) = fill(1,p.m),fill(1,p.m) #path,state : both are same
	
	function Base.iterate(p::Paths,state)
		if state ≠ fill(p.n,p.m)
			newstate = next(state,p.n)
			return newstate, newstate
		end
	end
	
	function next(path,n)
		k = length(path)
		 #start from the end and find the first element that can be updated by adding 
		while k >= 2 && (path[k]==n || path[k]+1 > path[k-1]+1)
			k-=1
		end
		path[k]+=1 #add the one and then reset the following elements
		for j = k+1: length(path)
			path[j] = max(path[j-1]-1,1)
		end
		return(path)
	end
	
	function allpaths(m,n)
		v = Vector{Int}[]
		paths = Paths(m,n)
		for p ∈ paths 
			push!(v,copy(p))
		end
		return(v)
	end
end

# ╔═╡ 269060a8-8cfa-46bd-a90a-b22f08ed4777
md"""

To iterate over an object, you need to define two iterate methods for your object: `iterate(x::MyType)` to get the first iteration, and `iterate(x::MyType, state)` for the rest. These methods should return nothing when the iterator is done, else a (result, state) tuple.

"""

# ╔═╡ 014bfea9-c7fb-4a57-b65f-a4c5c7fac38e
md""" 

For an 8x8 matrix the first path and last path are as follows.

"""

# ╔═╡ 72073148-68e6-4be5-824d-23de139a4fbc
first_path = fill(1,n)

# ╔═╡ 1c408f33-2be8-43c7-82b1-24eb5548f84e
last_path = fill(n,n)

# ╔═╡ 983bf28e-70da-4e11-9274-6b98ce270e20
begin
	paths = allpaths(n,n)
	numpaths = length(paths)
	md"There are $numpaths paths to check. "
end

# ╔═╡ 43be69fe-aed6-49ac-a9e4-7fa2bff74544
begin
	winnernum = argmin([sum( M1[i,p[i]] for i=1:n) for p∈paths])
	winner = paths[winnernum]
	winnertotal = sum( M1[i,winner[i]] for i=1:n);
end

# ╔═╡ a034032c-4ae0-4830-8ca3-ddf2a665939d
md"""
Path $(@bind whichpath Slider(1:20, show_value=true))
"""

# ╔═╡ 3eee2955-e8e0-4596-b15d-f1430a28a0d4
let
	
	path = paths[whichpath]
	values = [ M1[i,path[i]] for i=1:n]
	nv = length(values)
	thetitle = join([" $(values[i]) +" for i=1:nv-1 ]) * " $(values[end]) = $(sum(values))";
	
	
	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
	plot()
	for i=1:n, j=1:n
	   plot!(rectangle(1,1,i,j), opacity=.2, color=[:red,:white][1+rem(i+j,2) ])
	   
	end
	for i=1:n, j=1:n
	  annotate!((j+.5),n+2-(i+.5), M1[i,j])
	end
	
	# The winning path 
		for i = 1:n-1
		plot!([ winner[i+1]+.5, winner[i]+.5  ],[n-i+.5, n-i+1.5], color=RGB(1,.6,.6),  linewidth=4)
	end
	
	
	for i = 1:n-1
		plot!([ path[i+1]+.5, path[i]+.5  ],[n-i+.5, n-i+1.5], color=:black,  linewidth=4)
	end
	
	plot!(xlabel="winner total = $winnertotal", xguidefontcolor=RGB(1,.5,.5))

	
	for i=1:n,j=1:n
		plot!(rectangle(.4,.4,i+.3,j+.3), opacity=1, color=RGB(0,1,0), linewidth=0,fillcolor=[RGBA(1,.85,.85,.2),:white][1+rem(i+j,2)])
	end
	plot!(title=thetitle)
	plot!(legend=false, aspectratio=1, xlims=(1,n+1), ylims=(1,n+1), axis=nothing)
end

# ╔═╡ 75f4b8b9-d84b-4ce1-86d5-1f134ba377fe
md"""

### Dynamic Programming
"""

# ╔═╡ fb086920-8ee9-4971-9d08-a6b76d5b17de
md"""

Clearly, enumerating all seams and finding the optimal seam using brute force search is infeasible for larger matrices as the search space increases exponentially $$O(3^n)$$. We also see that we are recomputing the total energy for each path from scratch even though many paths differ only in the last few cells.


A better way to solve this problem is using Dynamic Programming.

"""



# ╔═╡ 3208e231-fa52-4927-82cc-6e7792c2a92d
md"""
Dynamic Programming involves computing the energy of the lowest energy seam that is possible when starting from a cell and recording it. This is a much simpler computation with a complexity of $$O(mn)$$
"""

# ╔═╡ 109a52c2-6459-436f-bbb4-9e0c8599df2f
md"""

The function `least_energy_matrix` computes this for a given matrix.
"""

# ╔═╡ 283d34f0-a99e-4fe9-865c-8f204b6f56c4
function least_energy_matrix(energies)
	result = copy(energies)
	m,n = size(energies)
	
	# your code here
	for i ∈ (m-1):-1:1
		for j ∈ 1:n
			
			if j==1
				successors = result[i+1,1:2]
			elseif j == n
				successors = result[i+1,j-1:j]
			else
				successors = result[i+1,j-1:j+1]
			end
			
			result[i,j] += minimum(successors)
		end
	end
	
	return result
end

# ╔═╡ e8b425df-3c1a-4b36-af69-13f70cf77504
M1

# ╔═╡ c522bcaa-2271-4a53-aa8c-3e7d5756389f
M2 = least_energy_matrix(M1)

# ╔═╡ 46af6252-638a-4657-bb98-e4c464a41970
md"""

The below function computes a minimum enery seam given a starting pixel.

"""

# ╔═╡ 795eb2c4-f37b-11ea-01e1-1dbac3c80c13
function seam_from_precomputed_least_energy(energies, starting_pixel::Int)
	least_energies = least_energy_matrix(energies)
	m, n = size(least_energies)
	
	seam = [starting_pixel]
	
	pixel = starting_pixel
	# Replace the following line with your code.
	for i in 1:(m-1)
		
		
		if pixel==1
			successors = OffsetVector(energies[i+1,1:2],0:1)
		elseif pixel == n
			successors = OffsetVector(energies[i+1,n-1:n],-1:0)
		else
			successors = OffsetVector(energies[i+1,pixel-1:pixel+1],-1:1)
		end
		
		# successors = energies[i+1,max(1,pixel-1):min(n,pixel+1)]
		
		winner_index = argmin(successors)
		next_pixel = pixel + winner_index
		
		push!(seam,next_pixel)
		pixel = next_pixel
		
	end
	
	return seam
	
end

# ╔═╡ f23bf7b8-d643-498b-9343-d618668fae17
optimal_seam = seam_from_precomputed_least_energy(M2,2)

# ╔═╡ aa6ee679-776d-485c-baaf-10c30869a9b4
md"""
The below function removes a given seam from the matrix.
"""

# ╔═╡ 90a22cc6-f327-11ea-1484-7fda90283797
function remove_in_each_row(img::Matrix, column_numbers::Vector)
	m, n = size(img)
	@assert m == length(column_numbers) # same as the number of rows

	local img′ = similar(img, m, n-1) # create a similar image with one column less

	for (i, j) in enumerate(column_numbers)
		img′[i, 1:j-1] .= @view img[i, 1:(j-1)]
		img′[i, j:end] .= @view img[i, (j+1):end]
		#view references the original object without making a copy
	end
	img′
end

# ╔═╡ 2b7fa0ab-462f-42fc-8592-9ab04815f8c7
remove_in_each_row(M1,optimal_seam)

# ╔═╡ 531eaff6-4b07-43e0-aae8-f67568a640c3
md"""

### Seam Carving on Images

"""

# ╔═╡ 0f271e1d-ae16-4eeb-a8a8-37951c70ba31
all_image_urls = [
	"https://wisetoast.com/wp-content/uploads/2015/10/The-Persistence-of-Memory-salvador-deli-painting.jpg" => "Salvador Dali — The Persistence of Memory (replica)",
	"https://i.imgur.com/4SRnmkj.png" => "Frida Kahlo — The Bride Frightened at Seeing Life Opened",
	"https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Hilma_af_Klint_-_Group_IX_SUW%2C_The_Swan_No._1_%2813947%29.jpg/477px-Hilma_af_Klint_-_Group_IX_SUW%2C_The_Swan_No._1_%2813947%29.jpg" => "Hilma Klint - The Swan No. 1",
	"https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Piet_Mondriaan%2C_1930_-_Mondrian_Composition_II_in_Red%2C_Blue%2C_and_Yellow.jpg/300px-Piet_Mondriaan%2C_1930_-_Mondrian_Composition_II_in_Red%2C_Blue%2C_and_Yellow.jpg" => "Piet Mondriaan - Composition with Red, Blue and Yellow",
]

# ╔═╡ 6dabe5e2-c851-4a2e-8b07-aded451d8058
md"""


 $(@bind image_url Select(all_image_urls))

Maximum image size: $(@bind max_height_str Select(string.([50,100,200,500]))) pixels. _(Using a large image might lead to long runtimes in the later exercises.)_
"""

# ╔═╡ 0d144802-f319-11ea-0028-cd97a776a3d0
img_original = load(download(image_url));

# ╔═╡ 22a5f4d6-e7d5-4ece-aa08-4a71f3aedd48
img_original

# ╔═╡ 0e9dbfca-3559-48d1-a7dd-0e74a35d76a2
img = imresize(img_original,ratio=0.2)

# ╔═╡ 9cc76c31-c285-4a1e-8eed-9e06957cb50a
e2 = energy2(img)

# ╔═╡ 5fccc7cc-f369-11ea-3b9e-2f0eca7f0f0e
md"""
A slightly modified version of the energy function is defined below.
"""

# ╔═╡ e9402079-713e-4cfd-9b23-279bd1d540f6
energy(∇x, ∇y) = sqrt.(∇x.^2 .+ ∇y.^2)

# ╔═╡ 6f37b34c-f31a-11ea-2909-4f2079bf66ec
function energy(img)
	∇y = convolve(brightness.(img), Kernel.sobel()[1])
	∇x = convolve(brightness.(img), Kernel.sobel()[2])
	energy(∇x, ∇y)
end

# ╔═╡ f721b866-509f-4a8f-b190-d702cd38d1a9
e1 = energy(img)

# ╔═╡ 51df0c98-f3c5-11ea-25b8-af41dc182bac
begin
	img, seam_from_precomputed_least_energy
	md"Compute shrunk image: $(@bind shrink_bottomup CheckBox())"
end

# ╔═╡ 946b69a0-f3a2-11ea-2670-819a5dafe891
if !@isdefined(seam_from_precomputed_least_energy)
	not_defined(:seam_from_precomputed_least_energy)
end

# ╔═╡ 14b3494d-3045-4315-965a-b52251866beb
md"""

## Conclusion

Julia is a high performance language that is already widely used in cutting edge scientific computing and high finance. It has the potential to be the pre-demoninant language for DS and ML in the years to come. However in popularity it continues to lag behind many languages. See [here](https://pypl.github.io/PYPL.html)



"""

# ╔═╡ 7e92cacd-20ec-4e57-bbd4-4b244bc8c4e6
md"""
## Resources
"""

# ╔═╡ c797cd21-7f8d-40f8-a5c7-2aab4fb19fd1
md"""

This introductory course at MIT is a great way to get started with Julia. [https://computationalthinking.mit.edu/Spring21/](https://computationalthinking.mit.edu/Spring21/)

The seam carving example above is drawn from the above course.

"""

# ╔═╡ 6b4d6584-f3be-11ea-131d-e5bdefcc791b
md"## Function library

Just some helper functions used in the notebook."

# ╔═╡ ef88c388-f388-11ea-3828-ff4db4d1874e
function mark_path(img, path)
	img′ = RGB.(img) # also makes a copy
	m = size(img, 2)
	for (i, j) in enumerate(path)
		if size(img, 2) > 50
			# To make it easier to see, we'll color not just
			# the pixels of the seam, but also those adjacent to it
			for j′ in j-1:j+1
				img′[i, clamp(j′, 1, m)] = RGB(1,0,1)
			end
		else
			img′[i, j] = RGB(1,0,1)
		end
	end
	img′
end

# ╔═╡ 437ba6ce-f37d-11ea-1010-5f6a6e282f9b
function shrink_n(min_seam::Function, img::Matrix{<:Colorant}, n, imgs=[];
		show_lightning=true,
	)
	
	n==0 && return push!(imgs, img)

	e = energy(img)
	seam_energy(seam) = sum(e[i, seam[i]]  for i in 1:size(img, 1))
	_, min_j = findmin(map(j->seam_energy(min_seam(e, j)), 1:size(e, 2)))
	min_seam_vec = min_seam(e, min_j)
	img′ = remove_in_each_row(img, min_seam_vec)
	if show_lightning
		push!(imgs, mark_path(img, min_seam_vec))
	else
		push!(imgs, img′)
	end
	shrink_n(min_seam, img′, n-1, imgs; show_lightning=show_lightning)
end

# ╔═╡ 51e28596-f3c5-11ea-2237-2b72bbfaa001
if shrink_bottomup
	local n = min(40, size(img, 2))
	bottomup_carved = shrink_n(seam_from_precomputed_least_energy, img, n)
	md"Shrink by: $(@bind bottomup_n Slider(1:n, show_value=true))"
end

# ╔═╡ 0a10acd8-f3c6-11ea-3e2f-7530a0af8c7f
if shrink_bottomup
	bottomup_carved[bottomup_n]
end

# ╔═╡ 6bdbcf4c-f321-11ea-0288-fb16ff1ec526
function decimate(img, n)
	img[1:n:end, 1:n:end]
end

# ╔═╡ 00115b6e-f381-11ea-0bc6-61ca119cb628
bigbreak = html"<br><br><br><br><br>";

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
ImageTransformations = "02fcd773-0e25-5acc-982a-7f6622650795"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
BenchmarkTools = "~1.2.0"
ImageFiltering = "~0.6.21"
ImageTransformations = "~0.8.13"
Images = "~0.24.1"
OffsetArrays = "~1.10.7"
Plots = "~1.22.6"
PlutoUI = "~0.7.16"
TestImages = "~1.6.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "61adeb0823084487000600ef8b1c00cc2474cd47"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "8d954297bc51cc64f15937c2093799c3617b73e4"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.10.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "45efb332df2e86f2cb2e992239b6267d97c9e0b6"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "6d1c23e740a586955645500bbec662476204a52c"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.1"

[[CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "70a0cfd9b1c86b0209e38fbfe6d8231fd606eeaf"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.1"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "d189c6d2004f63fd3c91748c458b09f26de0efaa"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.61.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cafe0823979a5c9bff86224b3b8de29ea5a44b2e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.61.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "f6532909bf3d40b308a0f360b6a0e626c0e263a8"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.1"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageAxes]]
deps = ["AxisArrays", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "794ad1d922c432082bc1aaa9fa8ffbd1fe74e621"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.9"

[[ImageContrastAdjustment]]
deps = ["ColorVectorSpace", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "2e6084db6cccab11fe0bc3e4130bd3d117092ed9"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.7"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageDistances]]
deps = ["ColorVectorSpace", "Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "6378c34a3c3a216235210d19b9f495ecfff2f85f"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.13"

[[ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "13c826abd23931d909e4c5538643d9691f62a617"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.8"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ea2b6fd947cdfc43c6b8c15cff982533ec1f72cd"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+0"

[[ImageMetadata]]
deps = ["AxisArrays", "ColorVectorSpace", "ImageAxes", "ImageCore", "IndirectArrays"]
git-tree-sha1 = "ae76038347dc4edcdb06b541595268fca65b6a42"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.5"

[[ImageMorphology]]
deps = ["ColorVectorSpace", "ImageCore", "LinearAlgebra", "TiledIteration"]
git-tree-sha1 = "68e7cbcd7dfaa3c2f74b0a8ab3066f5de8f2b71d"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.2.11"

[[ImageQualityIndexes]]
deps = ["ColorVectorSpace", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1198f85fa2481a3bb94bf937495ba1916f12b533"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.2.2"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageCore", "OffsetArrays", "Requires", "StackViews"]
git-tree-sha1 = "832abfd709fa436a562db47fd8e81377f72b01f9"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.1"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e4cc551e4295a5c96545bb3083058c24b78d4cf0"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.13"

[[Images]]
deps = ["AxisArrays", "Base64", "ColorVectorSpace", "FileIO", "Graphics", "ImageAxes", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageShow", "ImageTransformations", "IndirectArrays", "OffsetArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "8b714d5e11c91a0d945717430ec20f9251af4bd2"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.24.1"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "85e3436b18980e47604dd0909e37e2f066f54398"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.10"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "98f59ff3639b3d9485a03a72f3ab35bab9465720"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.6"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "ba43b248a1f04a9667ca4a9f782321d9211aa68e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.6"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "4c8a7d080daca18545c56f1cac28710c362478f3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.16"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rotations]]
deps = ["LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "2ed8d8a16d703f900168822d83699b8c3c1a5cd8"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.0.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "793793f1df98e3d7d554b65a107e9c9a6399a6ed"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.7.0"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StringDistances]]
deps = ["Distances"]
git-tree-sha1 = "a4c05337dfe6c4963253939d2acbdfa5946e8e31"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.10.0"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "db28237376a6b7ae9c9fe05880ece0ab8bb90b75"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.6.1"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "945b8d87c5e8d5e34e6207ee15edb9d11ae44716"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.3"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9398e8fefd83bde121d5127114bd3b6762c764a6"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─165932eb-2992-4580-947d-33a861d9007e
# ╟─2341517a-93e8-4c8c-afcc-87537776c09c
# ╟─467f60dc-26c0-411b-a548-0547f4ec4217
# ╟─2bab0ae8-3283-48d0-a17c-7a1b812e71c8
# ╟─6e0220d1-a566-4a2c-9a4b-4a5f8c55cdfb
# ╟─ff9a7a8f-160b-4438-a327-1cd672075725
# ╟─cafc8053-4ca7-4105-941b-6d8774366e21
# ╟─17603d7f-d337-4b6b-94c1-d304c951dabc
# ╟─045b7478-32a1-4d9e-b1ab-2b24a2d97601
# ╠═7eaa9450-1e09-11ec-175c-41430c2f547f
# ╠═6bb7a060-effe-11eb-02ce-c704b0dccd0c
# ╟─2d62de89-d56e-4483-abc4-53286fabff65
# ╟─fc593ffa-6bc1-4ac9-8d63-3475ddf85067
# ╟─6fec836e-b95f-4fb7-950f-0f8703662e0b
# ╠═b44a8d8d-fbef-4a17-9154-4b9bc76c2e05
# ╠═8e9875cb-0186-44ff-92e8-709684b64dae
# ╠═86e38c22-4fd8-4b8a-8fd3-468c971925f9
# ╠═2596db79-9244-4015-bfc3-c9444ca0d9c5
# ╠═77949b67-42ad-4567-9391-c2f384bc733a
# ╠═52edb6b6-ad57-47df-8a79-68f4003c7972
# ╟─5d31ad64-d7c1-42bb-a1ba-fc3add097ad7
# ╟─f37a2cbf-690d-4161-8dd3-c250760d8fa8
# ╟─64e3e5a3-be0f-4c8f-83f3-ceb093cd2f03
# ╟─2698e5c2-f3c0-40f3-9cd9-239e600a17ec
# ╠═38177084-bbb1-4f75-886c-89aa6686ed47
# ╠═5f55eb45-da6c-4ecc-941f-68ba9fcbd613
# ╠═bd1f8933-0791-43dd-a03d-83ba4a2cf6be
# ╟─518820ba-9254-4750-9675-eb5b369b6908
# ╟─12de93bb-1811-44f8-af21-3595360a0b0d
# ╠═dd22d4ef-822e-4272-a71d-650ff47eb32a
# ╟─13c210b7-6d90-4a3a-a088-ecbd069f3de5
# ╠═40daf36b-2de3-4f27-978b-a57a6492d0c5
# ╠═a5316ebc-9663-45a5-99b6-dfac03d2a1d5
# ╠═6d631da4-e32b-407b-85c4-a3dabe37e113
# ╠═fe365df6-f021-4a4f-9699-28d0672e169c
# ╟─3b2ad486-5d37-4f03-840c-85ebcba3096b
# ╠═261acfba-26ca-4f62-82c3-8beea15785ab
# ╠═c8a7adb0-6ea4-41bd-84b3-5dcb1f02dd7f
# ╟─5f73bd5d-f247-44e9-a1f6-3e2538e44717
# ╟─ee3f88f8-8a06-43fc-9fb7-a337d298c6b8
# ╠═42244456-dd4a-4f27-8d78-fd1a07dfaa29
# ╠═f424daaa-753c-46dc-96ea-3dc7dc6c600f
# ╟─35a2e4a0-d3a9-4d71-a96a-79983239f4de
# ╠═10f9b836-3d37-490b-8e10-02a0995e9e51
# ╠═33c5b3ec-912e-4f9d-9464-737b4330c283
# ╟─dfd32582-25da-4542-bcaa-1c1e0c508a75
# ╠═5e4da1c5-7cf0-40b7-a383-1caabe45e1ff
# ╟─79e08671-7bad-4aa4-898b-a2e5c72f886f
# ╠═57cb5c8d-52e2-4af7-9519-00feec2aade6
# ╠═6c2b94cb-0ed3-4316-a3a7-05aab39ba9c9
# ╟─0ca86629-9aea-482a-af01-f159d06ee5a6
# ╟─fc7c29ce-67bb-4814-ac46-3fd9c87ba7bd
# ╠═4f1036b4-ad87-4fa5-aa93-7ea5f29aa56d
# ╟─ad2a5d50-7757-4e67-8335-a9cfa8677943
# ╠═6ac64b90-b7a8-4db0-8976-8bb79ecf915f
# ╠═f60287e0-fe3d-4c5c-8084-dcc258afc41a
# ╠═e1e0e955-ec30-4f11-9f59-7025ecf3c86c
# ╟─1439a19f-1f85-4407-9790-7732c35bde64
# ╟─89c51fe2-1023-45e9-a0b5-dba8b165b3f7
# ╠═66fb8c0a-b050-4a75-9f55-151eaca0c836
# ╠═debd6aed-c218-4ae5-a97d-7afdfe9943ca
# ╠═cdef8039-fa96-4f79-935f-8050c047c858
# ╟─c406e5b1-d81d-49e0-8401-4369eb2714c5
# ╟─74a2b652-e68d-4d03-ac66-4dd039bcf418
# ╟─4b6ab118-31ff-4d33-bbb1-532c2bbac8cc
# ╠═8f64b69b-e2ac-4831-b874-6dd8aefac417
# ╠═5e3a09a2-a64d-4ded-985c-3067423cae57
# ╠═2bea3f87-dd6e-411a-8088-9e7e19c10da4
# ╠═3ed9bfc8-c59d-4aa2-a614-3844530e7b13
# ╟─3ca46cf8-6b70-4422-a264-3fe1b6658503
# ╟─0fa9affd-b887-459d-a3a3-11ce0087ec19
# ╟─0d605a3e-f898-46a7-9932-085a48287cb4
# ╟─97be249a-3105-4ea6-81a3-2077ef2ad242
# ╠═714bfd1d-0a0b-4a62-bdfe-1d3be6d9e878
# ╠═7e0f5420-d0dc-4b1d-8eb2-8962996eb483
# ╟─8d463199-243f-4f29-95ea-45ffd1509bd0
# ╟─03933a38-a1a1-4d37-acaf-2a49a4eaa334
# ╠═e2288fca-ff98-4702-b159-eb2605abe805
# ╠═152cc6f4-684a-4458-a120-49158668be52
# ╟─3741074c-46e9-4455-80ae-f4d15978030c
# ╠═1dd22345-89f7-4681-a1d8-236f3198eaba
# ╟─e390d90e-73de-4e37-a9e4-accf405f15a2
# ╠═70d7ea26-8701-4d41-b34b-4f8065092534
# ╟─376ad3f6-7f95-4167-b2be-fdc460419ee9
# ╠═a5122b62-be88-494a-9cfe-09603b308b9e
# ╠═5cf9beed-6864-4c3b-863b-947fb8c1e14f
# ╟─d2fc48ec-8b9e-42a2-af9b-f3a7617377de
# ╠═6c7e4b54-f318-11ea-2055-d9f9c0199341
# ╠═89adfdaa-376e-4d5b-a0d0-08d84b0dd810
# ╟─ae1c488f-e4c6-4301-8998-e1b392869a56
# ╠═cb72bf7f-c98a-42a8-8c32-8b1bccdffb58
# ╠═f721b866-509f-4a8f-b190-d702cd38d1a9
# ╠═9cc76c31-c285-4a1e-8eed-9e06957cb50a
# ╟─f404877d-ddce-4586-83f7-a0298a8c1c7b
# ╟─a74c03a2-eae7-4504-a63b-5395d794c983
# ╟─eb20bc08-ed6e-4dde-a69a-4c29ff4d6363
# ╟─cec83331-ed7f-4c5d-b5b9-f0600929e31b
# ╟─157d45c9-a3e0-4d46-bb19-1aa8d4955f45
# ╠═05654405-1991-48f8-bb8c-76d61596deec
# ╟─9a6a0b68-6eaa-4d92-9535-ffd5c9e6446f
# ╠═6b1ef6b2-2f0b-42e9-b87c-759e8b200543
# ╟─8b19990f-e55e-4255-8921-d202c46f9569
# ╟─269060a8-8cfa-46bd-a90a-b22f08ed4777
# ╟─014bfea9-c7fb-4a57-b65f-a4c5c7fac38e
# ╠═72073148-68e6-4be5-824d-23de139a4fbc
# ╠═1c408f33-2be8-43c7-82b1-24eb5548f84e
# ╟─983bf28e-70da-4e11-9274-6b98ce270e20
# ╠═43be69fe-aed6-49ac-a9e4-7fa2bff74544
# ╠═a034032c-4ae0-4830-8ca3-ddf2a665939d
# ╟─3eee2955-e8e0-4596-b15d-f1430a28a0d4
# ╟─75f4b8b9-d84b-4ce1-86d5-1f134ba377fe
# ╟─fb086920-8ee9-4971-9d08-a6b76d5b17de
# ╟─3208e231-fa52-4927-82cc-6e7792c2a92d
# ╟─109a52c2-6459-436f-bbb4-9e0c8599df2f
# ╟─283d34f0-a99e-4fe9-865c-8f204b6f56c4
# ╠═e8b425df-3c1a-4b36-af69-13f70cf77504
# ╠═c522bcaa-2271-4a53-aa8c-3e7d5756389f
# ╟─46af6252-638a-4657-bb98-e4c464a41970
# ╠═795eb2c4-f37b-11ea-01e1-1dbac3c80c13
# ╠═f23bf7b8-d643-498b-9343-d618668fae17
# ╟─aa6ee679-776d-485c-baaf-10c30869a9b4
# ╠═90a22cc6-f327-11ea-1484-7fda90283797
# ╠═2b7fa0ab-462f-42fc-8592-9ab04815f8c7
# ╟─531eaff6-4b07-43e0-aae8-f67568a640c3
# ╠═0f271e1d-ae16-4eeb-a8a8-37951c70ba31
# ╟─6dabe5e2-c851-4a2e-8b07-aded451d8058
# ╠═0d144802-f319-11ea-0028-cd97a776a3d0
# ╠═22a5f4d6-e7d5-4ece-aa08-4a71f3aedd48
# ╠═0e9dbfca-3559-48d1-a7dd-0e74a35d76a2
# ╟─5fccc7cc-f369-11ea-3b9e-2f0eca7f0f0e
# ╠═e9402079-713e-4cfd-9b23-279bd1d540f6
# ╠═6f37b34c-f31a-11ea-2909-4f2079bf66ec
# ╠═51df0c98-f3c5-11ea-25b8-af41dc182bac
# ╠═51e28596-f3c5-11ea-2237-2b72bbfaa001
# ╟─0a10acd8-f3c6-11ea-3e2f-7530a0af8c7f
# ╟─946b69a0-f3a2-11ea-2670-819a5dafe891
# ╟─14b3494d-3045-4315-965a-b52251866beb
# ╟─7e92cacd-20ec-4e57-bbd4-4b244bc8c4e6
# ╟─c797cd21-7f8d-40f8-a5c7-2aab4fb19fd1
# ╟─6b4d6584-f3be-11ea-131d-e5bdefcc791b
# ╟─437ba6ce-f37d-11ea-1010-5f6a6e282f9b
# ╟─ef88c388-f388-11ea-3828-ff4db4d1874e
# ╟─6bdbcf4c-f321-11ea-0288-fb16ff1ec526
# ╟─00115b6e-f381-11ea-0bc6-61ca119cb628
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
