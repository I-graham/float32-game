use winit::event;
use winit::event::Event;

use std::collections::HashMap;

use cgmath::Rotation3;

mod render;
use render::*;

const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 0.5, 0.0,
	0.0, 0.0, 0.5, 1.0,
);

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
				wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::STORAGE,
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

		{
			let (verts, indices) = Vertex::load_mesh("data/floater.obj", [0.8,0.8,0.8], 0.05);

			let vertices : &[Vertex] = verts.as_slice();

			let indices : &[u32] = indices.as_slice();

			scene.objects.insert("boat", Model::new(device, vertices, indices));
		}
		{
			let (verts, indices) = Vertex::load_mesh("data/sun.obj", [1.0,1.0,1.0], 0.05);

			let vertices : &[Vertex] = verts.as_slice();

			let indices : &[u32] = indices.as_slice();

			scene.objects.insert("sun", Model::new(device, vertices, indices));

			let model : &mut Model = scene.objects.get_mut("sun").unwrap();

			model.instances.push(Instance {
				position : [0.5; 3].into(),
				rotation : cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Deg(0.0)),
			}.to_matrix());
		}

		scene
	}

}

#[repr(C, packed)]
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
	win_size : [u32; 2],
	__align4 : [i32; 2],
}

impl Default for Uniform {
	fn default() -> Self {
		use cgmath::prelude::SquareMatrix;
		Uniform {
			time : 0,
			cam_position : [0.0,0.0,0.0],
			cam_proj : cgmath::Matrix4::identity(),
			light_position : [0.0, 7.0, 10.0],
			light_color : [0.8, 0.8, 0.8],
			win_size : [0; 2],
			__align0 : [0; 3],
			__align1 : [0; 1],
			__align2 : [0; 1],
			__align3 : [0; 1],
			__align4 : [0; 2],
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

#[repr(C, packed)]
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
	water_compute : (wgpu::ComputePipeline, wgpu::BindGroup),


	boats : Vec<Boat>,

}

impl GameState {

	const WATER_RES : usize = 250;

	fn generate_water(&mut self) {
		let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label : None,
		});

		{
			let mut cpass = encoder.begin_compute_pass();
			cpass.set_pipeline(&self.water_compute.0);
			cpass.set_bind_group(0, &self.water_compute.1, &[]);
			cpass.set_bind_group(1, &self.renderer.uniform_bg, &[]);
			cpass.dispatch(Self::WATER_RES as u32, Self::WATER_RES as u32, 1);
		}

		self.renderer.queue.submit(&[encoder.finish()]);
	}

	fn draw(&mut self) {

		match self.stage {
			Stage::Menu{..} => self.draw_menu(),
			Stage::Playing{..} => self.draw_game(),
		};

	}

	fn draw_menu(&mut self){

		if let Stage::Menu = self.stage {

			{

				let num_boats = self.boats.len();

				let boat_model = self.scene.objects.get_mut("boat").unwrap();

				boat_model.instances.reserve(num_boats);

				boat_model.instances.truncate(0);

				boat_model.instances.extend(self.boats.iter().map(|boat| boat.position.to_matrix()));

			}

			let frame = self.renderer.swap.get_next_texture().unwrap();

			let mut encoder = self.renderer.begin();

			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
					attachment : &self.renderer.msaa_texture.1,
					resolve_target : Some(&frame.view),
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
				screen_position: (self.win_state.win_size.width as f32 / 2.0f32, 0.9 * self.win_state.win_size.height as f32),
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

			{

				let num_boats = self.boats.len();

				let boat_model = self.scene.objects.get_mut("boat").unwrap();

				boat_model.instances.reserve(num_boats);

				boat_model.instances.truncate(0);

				boat_model.instances.extend(self.boats.iter().map(|boat| boat.position.to_matrix()));

			}

			let frame = self.renderer.swap.get_next_texture().unwrap();

			let mut encoder = self.renderer.begin();

			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
					attachment : &self.renderer.msaa_texture.1,
					resolve_target : Some(&frame.view),
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

		self.upload_uniform();

	}

	fn update_game(&mut self) {

		const SPEED : f32 = 0.05;

		self.camera.eye += cgmath::Vector3::new(
			(self.win_state.d_pressed as i8 - self.win_state.a_pressed as i8) as f32,
			(self.win_state.w_pressed as i8 - self.win_state.s_pressed as i8) as f32,
			(self.win_state.ctrl_down as i8 - self.win_state.shft_down as i8) as f32,
		) * SPEED;

		self.camera.target = self.camera.eye - cgmath::Vector3::unit_z();

		self.upload_uniform();
		self.generate_water();

	}

	fn upload_uniform(&mut self) {
		{
			self.uniforms.time += 1;
			self.uniforms.cam_position = self.camera.eye.into();
			self.uniforms.cam_proj = self.camera.build_view_projection_matrix();
			self.uniforms.win_size = [self.win_state.win_size.width, self.win_state.win_size.height];
		}

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

	fn resize(&mut self, dims : winit::dpi::PhysicalSize<u32>) {

		self.renderer.resize(dims);
		self.win_state.win_size = dims;
		self.camera.aspect = dims.width as f32 / dims.height as f32;

	}

}

async fn entry(event_loop : winit::event_loop::EventLoop<()>, window : winit::window::Window) {

	let sample_count = 2;

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

	let renderer = Renderer::new(&window, &[VERTEX_DESC], &[uniforms], sample_count).await;

	let mut state = {

		let mut scene : Scene = Scene::new(&renderer.device);

		let water_compute = {
			let bg_layout = renderer.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label : Some("Compute_layout"),
				bindings: &[wgpu::BindGroupLayoutEntry {
						binding : 0,
						visibility : wgpu::ShaderStage::COMPUTE,
						ty : wgpu::BindingType::StorageBuffer {
							dynamic : false,
							readonly : false,
						}
					},
					wgpu::BindGroupLayoutEntry {
						binding : 1,
						visibility : wgpu::ShaderStage::COMPUTE,
						ty : wgpu::BindingType::StorageBuffer {
							dynamic : false,
							readonly : false,
						}
					},
				]
			});

			let size = GameState::WATER_RES.pow(2) * std::mem::size_of::<Vertex>();

			let layout = &renderer.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
				bind_group_layouts : &[&bg_layout, &renderer.uniform_bgl],
			});

			let water_buff = renderer.device.create_buffer(&wgpu::BufferDescriptor {
				label : None,
				size : size as wgpu::BufferAddress,
				usage : wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::STORAGE,
			});

			let wave_num = 1;
			let wave_size = 4;
			let waves_len = wave_num * wave_size + 1;

			let mut waves_vec : Vec<f32> = vec![];

			use rand::distributions::{Distribution, Uniform};

			let range = Uniform::from(0.0f32..1.0f32);
			let mut rng = rand::thread_rng();

			waves_vec.push(0.0f32);
			waves_vec.extend([1.0, 1.0, 0.5, 1.5].iter());

			let char_slice = render::to_char_slice(waves_vec.as_mut_slice());
			char_slice[0..4].clone_from_slice(render::to_char_slice(&[wave_num as u32]));

			let waves_buff = renderer.device.create_buffer_with_data(
				char_slice,
				wgpu::BufferUsage::STORAGE
			);

			let bind_group = renderer.device.create_bind_group(&wgpu::BindGroupDescriptor{
				layout : &bg_layout,
				bindings : &[
					wgpu::Binding {
						binding : 0,
						resource : wgpu::BindingResource::Buffer {
							buffer : &water_buff,
							range : 0..size as wgpu::BufferAddress,
						}
					},
					wgpu::Binding {
						binding : 1,
						resource : wgpu::BindingResource::Buffer {
							buffer : &waves_buff,
							range : 0..(waves_len * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
						}
					},
				],
				label : None,
			});

			let module = &Renderer::shader_module(&renderer.device, std::path::Path::new("src/shaders/water.comp.spv"));

			let pipeline = renderer.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
				layout,
				compute_stage : wgpu::ProgrammableStageDescriptor {
					entry_point : "main",
					module,
				}
			});

			let mut indices = Vec::with_capacity((GameState::WATER_RES-1).pow(2));

			for i in 0..GameState::WATER_RES-1 {
				for j in 0..GameState::WATER_RES-1 {

					let x = i;
					let y = j;

					let to_coord = |x : usize, y : usize| -> u32 {
						(x * GameState::WATER_RES + y) as u32
					};

					indices.push(to_coord(x, y));
					indices.push(to_coord(x+1, y));
					indices.push(to_coord(x, y+1));

					indices.push(to_coord(x+1, y));
					indices.push(to_coord(x, y+1));
					indices.push(to_coord(x+1, y+1));

				}
			}

			let mesh = {

				let index_buff = renderer.device.create_buffer_with_data(
					render::to_char_slice(indices.as_slice()),
					wgpu::BufferUsage::INDEX,
				);

				(water_buff, index_buff, GameState::WATER_RES.pow(2) as u32, indices.len() as u32)

			};

			use cgmath::prelude::SquareMatrix;
			scene.objects.insert("water", Model {
				mesh,
				instances : vec![cgmath::Matrix4::identity()],
				bind_group : None,
			});

			(pipeline, bind_group)

		};

		let boats = vec![];

		GameState {
			stage : Stage::Menu,
			win_state : WinState::new(&renderer),
			uniforms,
			camera,
			renderer,
			scene,
			water_compute,
			boats,
		}
	};

	state.boats.push( Boat {
		position : Instance {
			position : [0.0, 0.0, 0.03].into(),
			rotation : cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_x(), cgmath::Deg(90.0)),
		}
	});

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

					event::WindowEvent::ScaleFactorChanged { new_inner_size : dims, .. } => state.resize(**dims),

					event::WindowEvent::Resized(dims) => state.resize(*dims),

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

	println!("vertex_buffer_size : {}", std::mem::size_of::<Vertex>());

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