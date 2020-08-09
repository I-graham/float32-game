pub struct Renderer {

	pub win_size     : winit::dpi::PhysicalSize<u32>,
	pub sample_count : u32,

	pub surface      : wgpu::Surface,
	pub adapter      : wgpu::Adapter,
	pub device       : wgpu::Device,
	pub queue        : wgpu::Queue,
	pub sc_desc      : wgpu::SwapChainDescriptor,
	pub swap         : wgpu::SwapChain,
	pub pipeline     : wgpu::RenderPipeline,
	pub uniform_bg   : wgpu::BindGroup,
	pub uniform_bgl  : wgpu::BindGroupLayout,
	pub uniform_buf  : wgpu::Buffer,
	pub instance_bgl : wgpu::BindGroupLayout,

	pub depth_buffer : (wgpu::Texture, wgpu::TextureView),
	pub msaa_texture : (wgpu::Texture, wgpu::TextureView),

}

impl Renderer {

	const DEPTH_FORMAT : wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

	pub async fn new<T>(win : &winit::window::Window, vertex_descs : &[wgpu::VertexBufferDescriptor<'_>], vertex_uniform : &[T], sample_count : u32) -> Self{

		let win_size = win.inner_size();

		let surface = wgpu::Surface::create(win);

		let adapter = wgpu::Adapter::request(
			&wgpu::RequestAdapterOptions {
				power_preference : wgpu::PowerPreference::LowPower,
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
			present_mode : wgpu::PresentMode::Immediate,

		};

		let swap = device.create_swap_chain(&surface, &sc_desc);

		let uniform_buf = device.create_buffer_with_data(
			to_char_slice(vertex_uniform),
			wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
		);

		let uniforms_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			bindings: &[
				wgpu::BindGroupLayoutEntry {
					binding : 0,
					visibility : wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT | wgpu::ShaderStage::COMPUTE,
					ty: wgpu::BindingType::UniformBuffer {
						dynamic : false,
					},
				},
			],
			label: Some("uniform_layout"),
		});

		let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
			layout: &uniforms_layout,
			bindings : &[
				wgpu::Binding {
					binding : 0,
					resource : wgpu::BindingResource::Buffer {
						buffer: &uniform_buf,
						range: 0..std::mem::size_of_val(vertex_uniform) as wgpu::BufferAddress,
					}
				},
				],
				label: Some("uniform_bg"),
		});

		let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			bindings : &[
				wgpu::BindGroupLayoutEntry {
					binding : 0,
					visibility : wgpu::ShaderStage::VERTEX,
					ty : wgpu::BindingType::StorageBuffer {
						dynamic : false,
						readonly : true,
					},
				},
			],
			label : Some("instances")
		});

		let depth_buffer = Self::create_depth_texture(&device, &sc_desc, sample_count);

		let pipeline = {

			let vertshader = Renderer::shader_module(&device, std::path::Path::new("src/shaders/default.vert.spv"));
			let fragshader = Renderer::shader_module(&device, std::path::Path::new("src/shaders/default.frag.spv"));

			let layout = &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				bind_group_layouts: &[&uniforms_layout, &instance_bgl],
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
						cull_mode : wgpu::CullMode::Back,
						depth_bias : 0,
						depth_bias_slope_scale : 0.0,
						depth_bias_clamp : 1.0,
					}),
					color_states : &[
						wgpu::ColorStateDescriptor {
							format : sc_desc.format,
							color_blend: wgpu::BlendDescriptor {
								src_factor: wgpu::BlendFactor::SrcAlpha,
								dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
								operation: wgpu::BlendOperation::Add,
							},
							alpha_blend: wgpu::BlendDescriptor {
								src_factor: wgpu::BlendFactor::One,
								dst_factor: wgpu::BlendFactor::One,
								operation: wgpu::BlendOperation::Add,
							},							write_mask : wgpu::ColorWrite::ALL,
						}
					],
					primitive_topology : wgpu::PrimitiveTopology::TriangleList,
					depth_stencil_state : Some( wgpu::DepthStencilStateDescriptor {
						format : Self::DEPTH_FORMAT,
						depth_write_enabled : true,
						depth_compare : wgpu::CompareFunction::LessEqual,
						stencil_front : wgpu::StencilStateFaceDescriptor::IGNORE,
						stencil_back : wgpu::StencilStateFaceDescriptor::IGNORE,
						stencil_read_mask : 0,
						stencil_write_mask : 0,
					}),
					vertex_state : wgpu::VertexStateDescriptor {
						index_format: wgpu::IndexFormat::Uint32,
						vertex_buffers: vertex_descs,
					},
					sample_count : sample_count,
					sample_mask: !0,
					alpha_to_coverage_enabled: false,

				}
			)

		};

		let msaa_texture = Self::create_multisampled_framebuffer(&device, &sc_desc, sample_count);

		Self {
			win_size,
			sample_count,
			surface,
			adapter,
			device,
			queue,
			sc_desc,
			swap,
			pipeline,
			uniform_bg,
			uniform_bgl : uniforms_layout,
			uniform_buf,
			depth_buffer,
			instance_bgl,
			msaa_texture
		}

	}

	pub fn resize(&mut self, size : winit::dpi::PhysicalSize<u32>) {
		self.win_size = size;
		self.sc_desc.width = size.width;
		self.sc_desc.height = size.height;
		self.swap = self.device.create_swap_chain(&self.surface, &self.sc_desc);
		self.depth_buffer = Self::create_depth_texture(&self.device, &self.sc_desc, self.sample_count);
		self.msaa_texture = Self::create_multisampled_framebuffer(&self.device, &self.sc_desc, self.sample_count);
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

	pub fn create_depth_texture(device : &wgpu::Device, sc_desc : &wgpu::SwapChainDescriptor, sample_count: u32) -> (wgpu::Texture, wgpu::TextureView) {

		let size = wgpu::Extent3d {
			width  : sc_desc.width,
			height : sc_desc.height,
			depth  : 1,
		};

		let desc = wgpu::TextureDescriptor {
			label : Some("Depth"),
			size,
			array_layer_count : 1,
			mip_level_count : 1,
			sample_count : sample_count,
			dimension : wgpu::TextureDimension::D2,
			format : Self::DEPTH_FORMAT,
			usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT
			| wgpu::TextureUsage::SAMPLED
			| wgpu::TextureUsage::COPY_SRC,
		};

		let texture = device.create_texture(&desc);
		let view = texture.create_default_view();

		(texture, view)

	}

	fn create_multisampled_framebuffer(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor, sample_count: u32) -> (wgpu::Texture, wgpu::TextureView) {
		let multisampled_texture_extent = wgpu::Extent3d {
			width: sc_desc.width,
			height: sc_desc.height,
			depth: 1,
		};
		let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
			size: multisampled_texture_extent,
			array_layer_count: 1,
			mip_level_count: 1,
			sample_count: sample_count,
			dimension: wgpu::TextureDimension::D2,
			format: sc_desc.format,
			usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
			label: None,
		};

		let texture = device.create_texture(multisampled_frame_descriptor);

		let view = texture.create_default_view();

		(texture, view)
	}

}

pub fn to_char_slice<T>(array : &[T]) -> &mut [u8] {

	let size = std::mem::size_of::<T>();

	let data_ptr = array.as_ptr() as *mut u8;

	unsafe { std::slice::from_raw_parts_mut(data_ptr, array.len() * size)}

}