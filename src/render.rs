pub struct Renderer {

	pub win_size    : winit::dpi::PhysicalSize<u32>,
	
	pub surface     : wgpu::Surface,
	pub adapter     : wgpu::Adapter,
	pub device      : wgpu::Device,
	pub queue       : wgpu::Queue,
	pub sc_desc     : wgpu::SwapChainDescriptor,
	pub swap        : wgpu::SwapChain,
	pub pipeline    : wgpu::RenderPipeline,
	pub uniform_bg  : wgpu::BindGroup,
	pub uniform_buf : wgpu::Buffer,

}

impl Renderer {

	pub async fn new<T>(win : &winit::window::Window, vertex_descs : &[wgpu::VertexBufferDescriptor<'_>], vertex_uniform : &[T]) -> Self{

		let win_size = win.inner_size();

		let surface = wgpu::Surface::create(win);

		let adapter = wgpu::Adapter::request(
			&wgpu::RequestAdapterOptions {
				power_preference : wgpu::PowerPreference::HighPerformance,
				compatible_surface : Some(&surface),
			},
			
			wgpu::BackendBit::PRIMARY,
		
		).await.unwrap();

		let (device, queue) = adapter.request_device(
			&wgpu::DeviceDescriptor {
				extensions: wgpu::Extensions {
					anisotropic_filtering: false,
				},
				limits : Default::default(),
			}
		).await;

		let sc_desc = wgpu::SwapChainDescriptor {

			usage        : wgpu::TextureUsage::OUTPUT_ATTACHMENT,
			format       : wgpu::TextureFormat::Bgra8UnormSrgb,
			width        : win_size.width,
			height       : win_size.height,
			present_mode : wgpu::PresentMode::Mailbox,

		};

		let swap = device.create_swap_chain(&surface, &sc_desc);
		
		let uniforms_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			bindings: &[
				wgpu::BindGroupLayoutEntry {
					binding : 0,
					visibility : wgpu::ShaderStage::VERTEX,
					ty: wgpu::BindingType::UniformBuffer {
						dynamic : false,
					},
				},
			],
			label: Some("uniform_layout"),
		});

		let uniform_buf = device.create_buffer_with_data(
			to_char_slice(vertex_uniform),
			wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
		);

		let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
			layout: &uniforms_layout,
			bindings : &[
				wgpu::Binding {
					binding: 0,
					resource: wgpu::BindingResource::Buffer {
						buffer: &uniform_buf,
						range: 0..std::mem::size_of_val(vertex_uniform) as wgpu::BufferAddress,
					}
				},
			],
			label: Some("uniform_vertices"),
		});
		
		let pipeline = {

			let vertshader = Renderer::shader_module(&device, std::path::Path::new("src/shaders/default.vert.spv"));
			let fragshader = Renderer::shader_module(&device, std::path::Path::new("src/shaders/default.frag.spv"));

			let layout = &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				bind_group_layouts: &[&uniforms_layout],
			});

			device.create_render_pipeline(
				&wgpu::RenderPipelineDescriptor {
					layout,
					vertex_stage: wgpu::ProgrammableStageDescriptor {
						module : &vertshader,
						entry_point: "main",
					},
					fragment_stage: Some (wgpu::ProgrammableStageDescriptor {
						module: &fragshader,
						entry_point: "main",
					}),
					rasterization_state: Some(wgpu::RasterizationStateDescriptor{
						front_face : wgpu::FrontFace::default(),
						cull_mode : wgpu::CullMode::None,
						depth_bias : 0,
						depth_bias_slope_scale : 0.0,
						depth_bias_clamp : 1.0,
					}),
					color_states : &[
						wgpu::ColorStateDescriptor {
							format : sc_desc.format,
							color_blend : wgpu::BlendDescriptor::REPLACE,
							alpha_blend : wgpu::BlendDescriptor::REPLACE,
							write_mask : wgpu::ColorWrite::ALL,
						}
					],
					primitive_topology : wgpu::PrimitiveTopology::TriangleList,
					depth_stencil_state : None,
					vertex_state : wgpu::VertexStateDescriptor {
						index_format: wgpu::IndexFormat::Uint32,
						vertex_buffers: vertex_descs,
					},
					sample_count : 1,
					sample_mask: !0,
					alpha_to_coverage_enabled: false,

				}
			)

		};

		Self {
			win_size,
			surface,
			adapter,
			device,
			queue,
			sc_desc,
			swap,
			pipeline,
			uniform_bg,
			uniform_buf,
		}

	}

	pub fn resize(&mut self, size : winit::dpi::PhysicalSize<u32>) {
		self.win_size = size;
		self.sc_desc.width = size.width;
		self.sc_desc.height = size.height;
		self.swap = self.device.create_swap_chain(&self.surface, &self.sc_desc);
	}

	pub fn shader_module(device : &wgpu::Device, name : &std::path::Path) -> wgpu::ShaderModule {

		let mut spirv = std::fs::File::open(name).unwrap();

		let data = wgpu::read_spirv(&mut spirv).unwrap();

		device.create_shader_module(&data)

	}

	pub fn begin(&self) -> wgpu::CommandEncoder{
	
		self.device.create_command_encoder(
			&wgpu::CommandEncoderDescriptor {
				label: Some("draw"),
			}
		)

	}

}

pub fn to_char_slice<T>(array : &[T]) -> &[u8] {
	
	let size = std::mem::size_of::<T>();

	let data_ptr = array.as_ptr() as *const u8;

	unsafe { std::slice::from_raw_parts(data_ptr, array.len() * size)}

}