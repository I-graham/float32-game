use shaderc;
use std::io::prelude::*;

fn main() -> std::io::Result<()> {
	
	let mut compiler = shaderc::Compiler::new().unwrap();
	
	for maybe_file in std::fs::read_dir("src/shaders").unwrap() {

		let file = maybe_file.unwrap();

		let pathbuf = file.path();
		
		let path = pathbuf.as_path();
		
		let extension = path.extension().unwrap().to_str().unwrap();

		let kind = {
			if extension == "vert"
			{
				shaderc::ShaderKind::Vertex
			} else if extension == "frag" {
				shaderc::ShaderKind::Fragment
			} else {
				continue
			}
		};
		
		let mut new_shader_name = String::from(path.to_str().unwrap());
		new_shader_name.push_str(".spv");

		let mut new_shader = std::fs::File::create(new_shader_name).unwrap();
		
		let source = std::fs::read_to_string(path)?;
		
		let spirv = compiler.compile_into_spirv(source.as_str(), kind, path.to_str().unwrap(), "main", None).map_err(|e| println!("{}", e)).expect("");

		new_shader.write(spirv.as_binary_u8()).unwrap();

	}
	Ok(())
}