use winit::event;
use winit::event::Event;

use std::collections::HashMap;

mod render;
use render::*;

const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 0.5, 0.0,
	0.0, 0.0, 0.5, 1.0,
);

fn perlin(x : f32, y : f32, z : f32) -> f32 {

	#[derive(Copy, Clone)]
	struct Vec3 {
		x : f32,
		y : f32,
		z : f32,
	}

	static T : [i32; 8] = [0x15,0x38,0x32,0x2c,0x0d,0x13,0x07,0x2a];

	static mut A : [i32; 3] = [0; 3];

	let s = (x + y + z) / 3.0;
	let ijk = Vec3 { x: (x+s).floor(), y : (y+s).floor(), z : (z+s).floor() } ;

	let s = (ijk.x + ijk.y + ijk.z) / 6.0;

	let uvw = Vec3 { x : x - ijk.x + s, y : y - ijk.y + s, z : z - ijk.z + s };

	let hi = if uvw.x >= uvw.z { if uvw.x >= uvw.y { 0 } else { 1 } } else { if uvw.y >= uvw.z { 1 } else { 2 } };
	let lo = if uvw.x <  uvw.z { if uvw.x <  uvw.y { 0 } else { 1 } } else { if uvw.y <  uvw.z { 1 } else { 2 } };

	unsafe { A = [0; 3]; }

	return k(hi, uvw, ijk) + k(3 - hi - lo, uvw, ijk) + k(lo, uvw, ijk) + k(0, uvw, ijk);

	fn get_a(id : i32) -> i32 {
		unsafe {A[id as usize]}
	}

	fn b1(n : i32, b : i32) -> i32 {
		n>>b&1
	}

	fn b2(i : i32, j : i32, k : i32, b : i32) -> i32 {
		T[(b1(i,b)<<2 | b1(j,b)<<1 | b1(k,b)) as usize]
	}

	fn shuffle(i : i32, j : i32, k : i32) -> i32 {
		b2(i,j,k,0) + b2(j,k,i,1) + b2(k,i,j,2) + b2(i,j,k,3) + b2(j,k,i,4) + b2(k,i,j,5) + b2(i,j,k,6) + b2(j,k,i,7)
	}

	fn k(a : i32, uvw : Vec3, ijk : Vec3) -> f32 {
		let s : f32 = (get_a(0)+get_a(1)+get_a(2)) as f32 / 6.0;

		let x : f32 = uvw.x - get_a(0) as f32 + s;
		let y : f32 = uvw.y - get_a(1) as f32 + s;
		let z : f32 = uvw.z - get_a(2) as f32 + s;
		let t : f32 = 0.6 - x * x - y * y - z * z;

		let h : i32 = shuffle(ijk.x as i32 + get_a(0), ijk.y as i32 + get_a(1), ijk.z as i32 + get_a(2));

		unsafe { A[a as usize] += 1 };

		if t < 0.0 {
			return 0.0;
		}

		let b5 : i32 = h>>5 & 1;
		let b4 : i32 = h>>4 & 1;
		let b3 : i32 = h>>3 & 1;
		let b2 : i32 = h>>2 & 1;
		let b  : i32 = h & 3;

		let p : f32 = if b==1 { x } else if b==2 { y } else { z };
		let q : f32 = if b==1 { y } else if b==2 { z } else { x };
		let r : f32 = if b==1 { z } else if b==2 { x } else { y };

		let p = if b5==b3 { -p } else { p };
		let q = if b5==b4 { -q } else { q };
		let r = if b5!=(b4^b3) { -r } else { r };

		let t = t * t;

		8.0 * t * t * (p + ( if b==0 { q+r } else { if b2==0 { q } else { r }}))

	}
}

const START_SIZE : winit::dpi::LogicalSize<f32> = winit::dpi::LogicalSize {
	width : 800.0,
	height : 600.0,
};

pub struct Camera {
	pub eye: cgmath::Point3<f32>,
	pub target: cgmath::Point3<f32>,
	pub up: cgmath::Vector3<f32>,
	pub aspect: f32,
	pub fovy: f32,
	pub znear: f32,
	pub zfar: f32,
}

impl Camera {

	fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {

		let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);

		let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

		return OPENGL_TO_WGPU_MATRIX * proj * view;
	}

}

struct Instance {
	position: cgmath::Vector3<f32>,
	rotation: cgmath::Quaternion<f32>,
}

impl Instance {
	fn to_matrix(&self) -> cgmath::Matrix4<f32> {
		cgmath::Matrix4::from_translation(self.position)
			* cgmath::Matrix4::from(self.rotation)
	}
}

struct Model {
	mesh : (wgpu::Buffer, wgpu::Buffer, u32, u32),
	instances : Vec<cgmath::Matrix4<f32>>,
	bind_group : Option<wgpu::BindGroup>,
}

impl Model {
	fn new(device : &wgpu::Device, vertices : &[Vertex], indices : &[u32]) -> Self {
		let mesh = {
			let vert_buff = device.create_buffer_with_data(
				render::to_char_slice(vertices),
				wgpu::BufferUsage::VERTEX,
			);

			let ind_buff = device.create_buffer_with_data(
				render::to_char_slice(indices),
				wgpu::BufferUsage::INDEX,
			);

			(vert_buff, ind_buff, vertices.len() as u32, indices.len() as u32)
		};

		let instances = vec![];

		Self {
			mesh,
			instances,
			bind_group : None,
		}

	}
}


struct Scene {
	objects : HashMap<&'static str, Model>,
}

impl Scene {
	fn new(device : &wgpu::Device) -> Self {
		let mut scene = Scene {
			objects : HashMap::new(),
		};

		let (verts, indices) = Vertex::load_mesh("data/floater.obj", [0.8,0.8,0.8], 0.5);

		let vertices : &[Vertex] = verts.as_slice();

		let indices : &[u32] = indices.as_slice();

		scene.objects.insert("boat", Model::new(device, vertices, indices));

		scene
	}

}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
struct Uniform {
	time : i32,
	__align0 : [i32; 3],
	cam_position : [f32; 3],
	__align1 : [i32; 1],
	cam_proj : cgmath::Matrix4<f32>,
	light_position : [f32; 3],
	__align2 : [i32; 1],
	light_color : [f32; 3],
	__align3 : [i32; 1],
}

impl Default for Uniform {
	fn default() -> Self {
		use cgmath::prelude::SquareMatrix;
		Uniform {
			time : 0,
			cam_position : [0.0,0.0,0.0],
			cam_proj : cgmath::Matrix4::identity(),
			light_position : [0.5, 0.5, 0.0],
			light_color : [1.0, 1.0, 1.0],
			__align0 : [0; 3],
			__align1 : [0; 1],
			__align2 : [0; 1],
			__align3 : [0; 1]
		}
	}
}

struct WinState {
	mouse_pos    : (f64, f64),
	mouse_down_l : bool,
	win_size     : winit::dpi::PhysicalSize<u32>,
	w_pressed    : bool,
	a_pressed    : bool,
	s_pressed    : bool,
	d_pressed    : bool,
	shft_down    : bool,
	ctrl_down    : bool,
}

impl WinState {
	pub fn new(renderer : &render::Renderer) -> Self {
		Self {
			win_size     : renderer.win_size,
			mouse_pos    : (0.0, 0.0),
			mouse_down_l : false,
			w_pressed    : false,
			a_pressed    : false,
			s_pressed    : false,
			d_pressed    : false,
			shft_down    : false,
			ctrl_down    : false,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
	position : [f32; 3],
	color : [f32; 3],
	normal : [f32; 3],
}

impl Vertex {

	fn load_mesh(filename : &str, color : [f32; 3], scale : f32) -> (Vec<Vertex>, Vec<u32>) {
		let obj = tobj::load_obj(filename, true).expect(format!("{}{}{}\n", "Missing asset :'", filename, "'").as_str());

		let (mut models, _materials) = obj;

		let model = models.remove(0);

		let mesh = model.mesh;

		let indices = mesh.indices;

		let length = mesh.positions.len() / 3;

		let mut verts : Vec<Vertex> = Vec::with_capacity(length);

		let mut position_iter = mesh.positions.iter().map(|x| scale * x);

		let mut normals_iter = mesh.normals.iter().map(|x| *x);

		for _ in 0..length {

			verts.push(
				Vertex {
					position : [position_iter.next().unwrap(), position_iter.next().unwrap(), position_iter.next().unwrap()],
					color,
					normal    : [normals_iter.next().unwrap(), normals_iter.next().unwrap(), normals_iter.next().unwrap()],
				}
			);
		}

		(verts, indices)

	}

}

const VERTEX_DESC : wgpu::VertexBufferDescriptor = wgpu::VertexBufferDescriptor {
	stride : std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
	step_mode : wgpu::InputStepMode::Vertex,
	attributes : &wgpu::vertex_attr_array![0 => Float3, 1 => Float3, 2 => Float3],
};

enum Stage {
	Menu,
	Playing,
}

struct Boat {
	position : Instance,
}

struct GameState {
	stage : Stage,
	win_state : WinState,
	uniforms : Uniform,
	camera : Camera,
	renderer : Renderer,
	scene : Scene,

	boats : Vec<Boat>,

	water : Vec<Vertex>,

}

impl GameState {

	const WATER_RES : usize = 40;
	const WATER_SCALE : f32 = 5.0;

	fn create_water(&mut self) {

		self.water.reserve(Self::WATER_RES.pow(2));

		let mut indices : Vec<u32> = Vec::with_capacity(6 * Self::WATER_RES.pow(2));

		for i in 0..Self::WATER_RES.pow(2) {

			let x = i / Self::WATER_RES;
			let y = i - x * Self::WATER_RES;

			let x_f = x as f32 / Self::WATER_RES as f32;
			let y_f = y as f32 / Self::WATER_RES as f32;

			let raw_height = perlin(Self::WATER_SCALE * x_f, Self::WATER_SCALE * y_f, 0.0) / 0.33;

			let height = raw_height / Self::WATER_RES as f32;

			let position = cgmath::Vector3::new(x_f, y_f, height);

			let normal = {

				let unit = 1.0 / Self::WATER_RES as f32;
				let x_cs = [Self::WATER_SCALE * (x_f - unit), Self::WATER_SCALE * (x_f + unit)];
				let y_cs = [Self::WATER_SCALE * (y_f - unit), Self::WATER_SCALE * (y_f + unit)];

				let corners = [
					cgmath::Vector3::new(
						x_cs[0],
						y_cs[0],
						perlin(x_cs[0], y_cs[0], 0.0)
					),
					cgmath::Vector3::new(
						x_cs[0],
						y_cs[1],
						perlin(x_cs[0], y_cs[1], 0.0)
					),
					cgmath::Vector3::new(
						x_cs[1],
						y_cs[0],
						perlin(x_cs[1], y_cs[0], 0.0)
					),
					cgmath::Vector3::new(
						x_cs[1],
						y_cs[1],
						perlin(x_cs[1], y_cs[1], 0.0)
					),
				];

				let mut normal = cgmath::Vector3::new(0.0,0.0,0.0);

				for i in 0..4 {

					let verts = [
						corners[i],
						corners[ (i + 1) % 4],
						position,
					];

					let edge1 = verts[0] - verts[2];
					let edge2 = verts[1] - verts[2];

					let perpendicular = edge1.cross(edge2);

					normal += perpendicular;

				}

				normal.z = 1.0;

				normal

			};

			self.water.push(Vertex {
				position : position.into(),
				color : [0.0,1.0,1.0].into(),
				normal : normal.into(),
			});

		}

		for i in 0..Self::WATER_RES-1 {
			for j in 0..Self::WATER_RES-1 {

				let x = i;
				let y = j;

				let to_coord = |x : usize, y : usize| -> u32 {
					(x * Self::WATER_RES + y) as u32
				};

				indices.push(to_coord(x, y));
				indices.push(to_coord(x+1, y));
				indices.push(to_coord(x, y+1));

				indices.push(to_coord(x+1, y));
				indices.push(to_coord(x, y+1));
				indices.push(to_coord(x+1, y+1));

			}
		}

		self.scene.objects.insert("water", Model::new(&self.renderer.device, self.water.as_slice(), indices.as_slice()));

		use cgmath::prelude::Rotation3;
		self.scene.objects.get_mut("water").unwrap().instances.push(
			Instance {
				position : [0.0, 0.0, 0.0].into(),
				rotation : cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Deg(0.0)),
			}.to_matrix()
		);

	}

	fn update_water(&mut self) {

		self.water.clear();

		for i in 0..Self::WATER_RES.pow(2) {

			let z_coord = self.uniforms.time as f32 / 60.0;

			let x = i / Self::WATER_RES;
			let y = i - x * Self::WATER_RES;

			let x_f = x as f32 / Self::WATER_RES as f32;
			let y_f = y as f32 / Self::WATER_RES as f32;

			let raw_height = perlin(Self::WATER_SCALE * x_f, Self::WATER_SCALE * y_f, z_coord) / 0.33;

			let height = raw_height / Self::WATER_RES as f32;

			let position = cgmath::Vector3::new(x_f, y_f, height);

			let normal = {

				let unit = 1.0 / Self::WATER_RES as f32;
				let x_cs = [Self::WATER_SCALE * (x_f - unit), Self::WATER_SCALE * (x_f + unit)];
				let y_cs = [Self::WATER_SCALE * (y_f - unit), Self::WATER_SCALE * (y_f + unit)];

				let corners = [
					cgmath::Vector3::new(
						x_cs[0],
						y_cs[0],
						perlin(x_cs[0], y_cs[0], z_coord)
					),
					cgmath::Vector3::new(
						x_cs[0],
						y_cs[1],
						perlin(x_cs[0], y_cs[1], z_coord)
					),
					cgmath::Vector3::new(
						x_cs[1],
						y_cs[0],
						perlin(x_cs[1], y_cs[0], z_coord)
					),
					cgmath::Vector3::new(
						x_cs[1],
						y_cs[1],
						perlin(x_cs[1], y_cs[1], z_coord)
					),
				];

				let mut normal = cgmath::Vector3::new(0.0,0.0,0.0);

				for i in 0..4 {

					let verts = [
						corners[i],
						corners[ (i + 1) % 4],
						position,
					];

					let edge1 = verts[0] - verts[2];
					let edge2 = verts[1] - verts[2];

					let perpendicular = edge1.cross(edge2);

					normal += perpendicular;

				}

				normal.z = 1.0;

				normal

			};

			self.water.push(Vertex {
				position : position.into(),
				color : [0.0,1.0,1.0].into(),
				normal : normal.into(),
			});

		}

		let mut model = self.scene.objects.get_mut("water").unwrap();

		let slice = render::to_char_slice(self.water.as_slice());

		model.mesh.0 = self.renderer.device.create_buffer_with_data(
			slice,
			wgpu::BufferUsage::VERTEX
		);

	}

	fn draw(&mut self) {

		match self.stage {
			Stage::Menu{..} => self.draw_menu(),
			Stage::Playing{..} => self.draw_game(),
		};

	}

	fn draw_menu(&mut self){

		if let Stage::Menu = self.stage {

			let frame = self.renderer.swap.get_next_texture().unwrap();
			{

				let num_boats = self.boats.len();

				let boat_model = self.scene.objects.get_mut("boat").unwrap();

				boat_model.instances.reserve(num_boats);

				boat_model.instances.truncate(0);

				boat_model.instances.extend(self.boats.iter().map(|boat| boat.position.to_matrix()));

			}

			let mut encoder = self.renderer.begin();

			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
					attachment : &frame.view,
					resolve_target : None,
					load_op: wgpu::LoadOp::Clear,
					store_op: wgpu::StoreOp::Store,
					clear_color: wgpu::Color {
						r: 0.0,
						g: 0.8,
						b: 1.0,
						a: 1.0,
					},
					}],
				depth_stencil_attachment: Some( wgpu::RenderPassDepthStencilAttachmentDescriptor {
					attachment : &self.renderer.depth_buffer.1,
					depth_load_op : wgpu::LoadOp::Clear,
					depth_store_op : wgpu::StoreOp::Store,
					clear_depth : 1.0,
					stencil_load_op : wgpu::LoadOp::Clear,
					stencil_store_op : wgpu::StoreOp::Store,
					clear_stencil : 0,
				}),
			});

			render_pass.set_pipeline(&self.renderer.pipeline);

			for model in self.scene.objects.values_mut().filter(|model| model.instances.len() != 0) {

				let instances = &model.instances;

				let instance_slice = render::to_char_slice(instances.as_slice());

				let instance_buff = self.renderer.device.create_buffer_with_data(
					instance_slice,
					wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::STORAGE_READ,
				);

				model.bind_group = Some(self.renderer.device.create_bind_group(&wgpu::BindGroupDescriptor{
					layout : &self.renderer.instance_bgl,
					bindings : &[
						wgpu::Binding {
							binding : 0,
							resource : wgpu::BindingResource::Buffer {
								buffer : &instance_buff,
								range : 0..instance_slice.len() as wgpu::BufferAddress,
							},
						}
				],
					label : None,
				}));

				let mesh = &model.mesh;

				let instances = &model.instances;

				render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

				render_pass.set_bind_group(1, model.bind_group.as_ref().unwrap(), &[]);

				render_pass.set_vertex_buffer(0, &mesh.0, 0, 0);

				render_pass.set_index_buffer(&mesh.1, 0, 0);

				render_pass.draw_indexed(0..mesh.3, 0, 0..instances.len() as u32);

			}

			drop(render_pass);

			use wgpu_glyph::{ab_glyph, GlyphBrushBuilder, Section, Text, Layout, HorizontalAlign};

			let font = ab_glyph::FontArc::try_from_slice(include_bytes!("../data/bahnschrift.ttf")).expect("unable to load font.");

			let mut brush = GlyphBrushBuilder::using_font(font).build(&self.renderer.device, self.renderer.sc_desc.format);

			let section = Section {
				screen_position: (self.win_state.win_size.width as f32 / 2.0f32, 1.8 * self.win_state.win_size.height as f32 / 2.0f32),
				text : vec![Text::new("Press space to begin").with_scale(70.0).with_color([0.0,0.0,0.0,1.0])],
				layout : Layout::default_single_line().h_align(HorizontalAlign::Center),
				..Section::default()
			};

			brush.queue(section);

			brush.draw_queued(
				&self.renderer.device,
				&mut encoder,
				&frame.view,
				self.renderer.sc_desc.width,
				self.renderer.sc_desc.height,
			).unwrap();

			self.renderer.queue.submit(&[encoder.finish()]);

		} else  { panic!("type mismatch!"); }
	}

	fn draw_game(&mut self) {

		if let Stage::Playing = &self.stage {

			let frame = self.renderer.swap.get_next_texture().unwrap();

			let mut encoder = self.renderer.begin();

			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
					attachment : &frame.view,
					resolve_target : None,
					load_op: wgpu::LoadOp::Clear,
					store_op: wgpu::StoreOp::Store,
					clear_color: wgpu::Color {
						r: 0.0,
						g: 0.8,
						b: 1.0,
						a: 1.0,
					},
					}],
				depth_stencil_attachment: Some( wgpu::RenderPassDepthStencilAttachmentDescriptor {
					attachment : &self.renderer.depth_buffer.1,
					depth_load_op : wgpu::LoadOp::Clear,
					depth_store_op : wgpu::StoreOp::Store,
					clear_depth : 1.0,
					stencil_load_op : wgpu::LoadOp::Clear,
					stencil_store_op : wgpu::StoreOp::Store,
					clear_stencil : 0,
				}),
			});

			render_pass.set_pipeline(&self.renderer.pipeline);

			for model in self.scene.objects.values_mut().filter(|model| model.instances.len() != 0) {

				let instances = &model.instances;

				let instance_slice = render::to_char_slice(instances.as_slice());

				let instance_buff = self.renderer.device.create_buffer_with_data(
					instance_slice,
					wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::STORAGE_READ,
				);

				model.bind_group = Some(self.renderer.device.create_bind_group(&wgpu::BindGroupDescriptor{
					layout : &self.renderer.instance_bgl,
					bindings : &[
						wgpu::Binding {
							binding : 0,
							resource : wgpu::BindingResource::Buffer {
								buffer : &instance_buff,
								range : 0..instance_slice.len() as wgpu::BufferAddress,
							},
						}
				],
					label : None,
				}));

				let mesh = &model.mesh;

				let instances = &model.instances;

				render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

				render_pass.set_bind_group(1, model.bind_group.as_ref().unwrap(), &[]);

				render_pass.set_vertex_buffer(0, &mesh.0, 0, 0);

				render_pass.set_index_buffer(&mesh.1, 0, 0);

				render_pass.draw_indexed(0..mesh.3, 0, 0..instances.len() as u32);

			}

			drop(render_pass);

			self.renderer.queue.submit(&[encoder.finish()]);

		} else  { panic!("type mismatch!"); }

	}

	fn update(&mut self) {
		match self.stage {
			Stage::Menu {..} => self.update_menu(),
			Stage::Playing {..} => self.update_game(),
		}
	}

	fn update_menu(&mut self) {

		self.uniforms.time += 1;
		self.upload_uniform();

	}

	fn update_game(&mut self) {

		self.uniforms.time += 1;

		const SPEED : f32 = 0.05;

		self.camera.eye += cgmath::Vector3::new(
			(self.win_state.d_pressed as i8 - self.win_state.a_pressed as i8) as f32,
			(self.win_state.w_pressed as i8 - self.win_state.s_pressed as i8) as f32,
			(self.win_state.ctrl_down as i8 - self.win_state.shft_down as i8) as f32,
		) * SPEED;

		self.update_water();

		//self.camera.target = self.camera.eye - cgmath::Vector3::unit_z();

		self.uniforms.cam_proj = self.camera.build_view_projection_matrix();

		self.upload_uniform();

	}

	fn upload_uniform(&self) {
		let mut encoder = self.renderer.begin();

		let new_uniform = self.renderer.device.create_buffer_with_data(
			render::to_char_slice(&[self.uniforms]),
			wgpu::BufferUsage::COPY_SRC,
		);

		encoder.copy_buffer_to_buffer(
			&new_uniform,
			0 as wgpu::BufferAddress,
			&self.renderer.uniform_buf,
			0 as wgpu::BufferAddress,
			std::mem::size_of::<Uniform>() as wgpu::BufferAddress,
		);

		self.renderer.queue.submit(&[encoder.finish()]);

	}

	fn input(&mut self, event : &winit::event::WindowEvent) {
		match event {
			event::WindowEvent::CursorMoved {
				position,
				..
			} => { self.win_state.mouse_pos = ((2.0 * position.x / self.win_state.win_size.width as f64) - 1.0, (-2.0 * position.y / self.win_state.win_size.height as f64) + 1.0); },

			#[allow(deprecated)]
			event::WindowEvent::KeyboardInput {
				input : event::KeyboardInput {
					virtual_keycode : Some(key),
					state,
					modifiers,
					..
				},
				..
			} => {

				self.win_state.ctrl_down = modifiers.ctrl();
				self.win_state.shft_down = modifiers.shift();

				match key {

					event::VirtualKeyCode::W => {
						self.win_state.w_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::A => {
						self.win_state.a_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::S => {
						self.win_state.s_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::D => {
						self.win_state.d_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::Space if *state == event::ElementState::Pressed => {

						if let Stage::Menu = &self.stage {
							self.stage = Stage::Playing;
						};

					},

					_ => {},
				}

			},

			event::WindowEvent::MouseInput {
				button,
				state,
				..
			} => {
				match button {

					event::MouseButton::Left => {
						self.win_state.mouse_down_l = *state == event::ElementState::Pressed;
					},

					_ => (),
				}
			},

			_ => (),

		};
	}
}

async fn entry(event_loop : winit::event_loop::EventLoop<()>, window : winit::window::Window) {

	let camera = Camera {
		eye    : (0.0, 0.0, 1.0).into(),
		target : (0.0, 0.0, 0.0).into(),
		up     : cgmath::Vector3::unit_y(),
		aspect : window.inner_size().width as f32 / window.inner_size().height as f32,
		fovy   : 45.0,
		znear  : 0.01,
		zfar   : 500.0,
	};

	let uniforms = Uniform {
		cam_proj : camera.build_view_projection_matrix(),
		time : 0,
		cam_position : camera.eye.into(),
		..Default::default()
	};

	let renderer = Renderer::new(&window, &[VERTEX_DESC], &[uniforms]).await;

	let mut state = {

		let scene : Scene = Scene::new(&renderer.device);

		let boats = vec![];

		let water = vec![];

		GameState {
			stage : Stage::Menu,
			win_state : WinState::new(&renderer),
			uniforms,
			camera,
			renderer,
			scene,
			boats,
			water,
		}
	};

	state.create_water();

/*	use cgmath::prelude::Rotation3;
	state.boats.push( Boat {
		position : Instance {
			position : [0.0, 0.0, -1.0].into(),
			rotation : cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Deg(0.0)),
		}
	});*/

	let mut fps = 0;
	let mut fps_timer = std::time::Instant::now();

	event_loop.run(move |event, _, control_flow| {

		match event {

			Event::WindowEvent {
				event,
				window_id
			} if window_id == window.id() => {
				match &event {

					event::WindowEvent::CloseRequested => *control_flow = winit::event_loop::ControlFlow::Exit,

					event::WindowEvent::ScaleFactorChanged { new_inner_size : dims, .. } => state.renderer.resize(**dims),

					event::WindowEvent::Resized(dims) => {

						state.renderer.resize(*dims);
						state.win_state.win_size = *dims;
					},

					event::WindowEvent::KeyboardInput { input : event::KeyboardInput {
						virtual_keycode : Some(key),
						state,
						..
					}, .. } => {
						match key {

							event::VirtualKeyCode::F11 if *state == event::ElementState::Released => {
								if let Some(_) = window.fullscreen() {
									window.set_fullscreen(None);
								} else {
									window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(window.primary_monitor())));
								}
							},

							event::VirtualKeyCode::Escape => {
								*control_flow = winit::event_loop::ControlFlow::Exit;
							},

							_ => {},

						}
					},

					_ => {},

				}

				state.input(&event);
			},

			Event::MainEventsCleared =>  {

				state.update();

				window.request_redraw();
				fps += 1;

				let now = std::time::Instant::now();

				let duration = now.duration_since(fps_timer).as_secs();

				if duration >= 1 {
					println!("fps : {}", fps);
					fps = 0;
					fps_timer = now;
				}

			},

			Event::RedrawRequested(_) => state.draw(),

			_ => (),

		};

	});

}

fn main() {

	let event_loop = winit::event_loop::EventLoop::new();

	let window = winit::window::WindowBuilder::new()
		.with_min_inner_size(START_SIZE)
		.with_title("Float32")
		.with_inner_size(START_SIZE)
		.build(&event_loop)
		.unwrap();

	window.set_cursor_icon(winit::window::CursorIcon::Default);

	#[cfg(target_arch = "wasm32")]
	{
		std::panic::set_hook(Box::new(console_error_panic_hook::hook));

		web_sys::window()
			.and_then(|win | win.document())
			.and_then(|doc | doc.document())
			.and_then(|body| body.append_child(&web_sys::Element::from(window.canvas())).ok()).expect("unable to create canvas to document body");

		wasm_bindgen_futures::spawn_local(entry(event_loop, window));
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		futures::executor::block_on(entry(event_loop, window));
	}
}