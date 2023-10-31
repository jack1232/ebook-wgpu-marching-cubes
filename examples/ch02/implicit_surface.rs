use bytemuck::cast_slice;
use cgmath::{Matrix, Matrix4, SquareMatrix};
use rand::{rngs::ThreadRng, Rng};
use std::{collections::HashMap, iter};
use wgpu::{util::DeviceExt, VertexBufferLayout};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu_simplified as ws;
use wgpu_marching_cubes::{colormap, marching_cubes_table};

fn create_color_data(colormap_name: &str) -> Vec<[f32; 4]> {
    let cdata = colormap::colormap_data(colormap_name);
    let mut data: Vec<[f32; 4]> = vec![];
    for i in 0..cdata.len() {
        data.push([cdata[i][0], cdata[i][1], cdata[i][2], 1.0]);
    }
    data
}

fn surface_type_map() -> HashMap<u32, String> {
    let mut surface_type = HashMap::new();
    surface_type.insert(0, String::from("Sphere"));
    surface_type.insert(1, String::from("Schwartz Surface"));
    surface_type.insert(2, String::from("Blobs"));
    surface_type.insert(3, String::from("Klein"));
    surface_type.insert(4, String::from("Torus"));
    surface_type.insert(5, String::from("Chmutov"));
    surface_type.insert(6, String::from("Gyroid"));
    surface_type.insert(7, String::from("Cube Sphere"));
    surface_type.insert(8, String::from("Ortho Circle"));
    surface_type.insert(9, String::from("Spider Cage"));
    surface_type.insert(10, String::from("Barth Sextic"));
    surface_type
}

fn get_surface_type(key: u32) -> String {
    let map = surface_type_map();
    map.get(&key).map(|s| s.to_string()).unwrap_or_default()
}

struct State {
    init: ws::IWgpuInit,
    pipeline: wgpu::RenderPipeline,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    uniform_buffers: Vec<wgpu::Buffer>,

    cs_pipelines: Vec<wgpu::ComputePipeline>,
    cs_vertex_buffers: Vec<wgpu::Buffer>,
    cs_index_buffer: wgpu::Buffer,
    cs_uniform_buffers: Vec<wgpu::Buffer>,
    cs_bind_groups: Vec<wgpu::BindGroup>,

    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    msaa_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    animation_speed: f32,
    rotation_speed: f32,

    resolution: u32,
    index_count: u32,

    surface_type: u32,
    colormap_direction: u32,
    colormap_reverse: u32,
    isolevel: f32,
    scale: f32,

    rng: ThreadRng,
    t0: std::time::Instant,
    random_shape_change: u32,
    fps_counter: ws::FpsCounter,
}

impl State {
    async fn new(window: &Window, sample_count: u32, resolution: u32, colormap_name: &str) -> Self {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 1024 * 1024 * 1024, //1024MB, defaulting to 128MB
            max_buffer_size: 1024 * 1024 * 1024,                 // 1024MB, defaulting to 256MB
            max_compute_invocations_per_workgroup: 512,          // dafaulting to 256
            ..Default::default()
        };
        let init = ws::IWgpuInit::new(&window, sample_count, Some(limits)).await;

        let resol = ws::round_to_multiple(resolution, 8);
        let marching_cube_cells = (resolution - 1) * (resolution - 1) * (resolution - 1);
        let vertex_count = 3 * 12 * marching_cube_cells;
        let vertex_buffer_size = 4 * vertex_count;
        let index_count = 15 * marching_cube_cells;
        let index_buffer_size = 4 * index_count;
        println!("resolution = {}", resol);

        let vs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("../ch01/shader_vert.wgsl"));
        let fs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("../ch01/shader_frag.wgsl"));

        let cs_value_file = include_str!("implicit_value.wgsl");
        let cs_func_file = include_str!("implicit_func.wgsl");
        let cs_surface_file = include_str!("implicit_surface.wgsl");

        let cs_value = init
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Value Shader"),
                source: wgpu::ShaderSource::Wgsl([cs_func_file, cs_value_file].join("\n").into()),
            });

        let cs_comp = init
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Surface Shader"),
                source: wgpu::ShaderSource::Wgsl([cs_func_file, cs_surface_file].join("\n").into()),
            });

        // uniform data
        let camera_position = (2.0, 2.0, 3.0).into();
        let look_direction = (0.0, 0.0, 0.0).into();
        let up_direction = cgmath::Vector3::unit_y();
        let light_direction = [-0.5f32, -0.5, -0.5];

        let (view_mat, project_mat, _) = ws::create_vp_mat(
            camera_position,
            look_direction,
            up_direction,
            init.config.width as f32 / init.config.height as f32,
        );

        // create vertex uniform buffers

        // model_mat and vp_mat will be stored in vertex_uniform_buffer inside the update function
        let vert_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Uniform Buffer"),
            size: 192,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // create light uniform buffer. here we set eye_position = camera_position
        let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let eye_position: &[f32; 3] = camera_position.as_ref();
        init.queue.write_buffer(
            &light_uniform_buffer,
            0,
            cast_slice(light_direction.as_ref()),
        );
        init.queue
            .write_buffer(&light_uniform_buffer, 16, cast_slice(eye_position));

        // set specular light color to white
        let specular_color: [f32; 3] = [1.0, 1.0, 1.0];
        init.queue.write_buffer(
            &light_uniform_buffer,
            32,
            cast_slice(specular_color.as_ref()),
        );

        // material uniform buffer
        let material_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // set default material parameters
        let material = [0.1f32, 0.7, 0.4, 30.0];
        init.queue
            .write_buffer(&material_uniform_buffer, 0, cast_slice(material.as_ref()));

        // uniform bind group for vertex shader
        let (vert_bind_group_layout, vert_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::VERTEX],
            &[vert_uniform_buffer.as_entire_binding()],
        );

        // uniform bind group for fragment shader
        let (frag_bind_group_layout, frag_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::FRAGMENT, wgpu::ShaderStages::FRAGMENT],
            &[
                light_uniform_buffer.as_entire_binding(),
                material_uniform_buffer.as_entire_binding(),
            ],
        );

        let vertex_buffer_layouts = [
            VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4], // pos
            },
            VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![1 => Float32x4], // norm
            },
            VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![2 => Float32x4], // col
            },
        ];

        let pipeline_layout = init
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&vert_bind_group_layout, &frag_bind_group_layout],
                push_constant_ranges: &[],
            });

        let mut ppl = ws::IRenderPipeline {
            vs_shader: Some(&vs_shader),
            fs_shader: Some(&fs_shader),
            pipeline_layout: Some(&pipeline_layout),
            vertex_buffer_layout: &vertex_buffer_layouts,
            ..Default::default()
        };
        let pipeline = ppl.new(&init);

        let msaa_texture_view = ws::create_msaa_texture_view(&init);
        let depth_texture_view = ws::create_depth_view(&init);

        // create compute pipeline for value
        let volume_elements = resol * resol * resol;
        let cs_value_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Value Buffer"),
            size: 4 * volume_elements as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_value_int_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compuet Value Integer Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_value_float_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compuet Value Float Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (cs_value_bind_group_layout, cs_value_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
            ],
            vec![
                wgpu::BufferBindingType::Storage { read_only: false },
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Uniform,
            ],
            &[
                cs_value_buffer.as_entire_binding(),
                cs_value_int_buffer.as_entire_binding(),
                cs_value_float_buffer.as_entire_binding(),
            ],
        );

        let cs_value_pipeline_layout =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Value Pipeline Layout"),
                    bind_group_layouts: &[&cs_value_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let cs_value_pipeline =
            init.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Value Pipeline"),
                    layout: Some(&cs_value_pipeline_layout),
                    module: &cs_value,
                    entry_point: "cs_main",
                });

        // create compute pipeline for implicit surface
        let cs_table_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Table STorage Buffer"),
            size: (marching_cubes_table::EDGE_TABLE.len() + marching_cubes_table::TRI_TABLE.len())
                as u64
                * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue.write_buffer(
            &cs_table_buffer,
            0,
            cast_slice(marching_cubes_table::EDGE_TABLE),
        );
        init.queue.write_buffer(
            &cs_table_buffer,
            marching_cubes_table::EDGE_TABLE.len() as u64 * 4,
            cast_slice(marching_cubes_table::TRI_TABLE),
        );

        let cs_position_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Position Buffer"),
            size: vertex_buffer_size as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_normal_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Normal Buffer"),
            size: vertex_buffer_size as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_color_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Color Buffer"),
            size: vertex_buffer_size as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_index_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Index Buffer"),
            size: index_buffer_size as u64,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        //let indirect_array = [500u32, 0, 0, 0];
        let cs_indirect_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Indirect Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cdata = create_color_data(colormap_name);
        let cs_colormap_buffer =
            init.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Compute Colormap Uniform Buffer"),
                    contents: bytemuck::cast_slice(&cdata),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let cs_int_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Integer uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_float_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Float uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (cs_bind_group_layout, cs_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
            ],
            vec![
                wgpu::BufferBindingType::Storage { read_only: true }, // marching table
                wgpu::BufferBindingType::Storage { read_only: true }, // value buffer
                wgpu::BufferBindingType::Storage { read_only: false }, // position
                wgpu::BufferBindingType::Storage { read_only: false }, // normal
                wgpu::BufferBindingType::Storage { read_only: false }, // color
                wgpu::BufferBindingType::Storage { read_only: false }, // indices
                wgpu::BufferBindingType::Storage { read_only: false }, // indirect params
                wgpu::BufferBindingType::Storage { read_only: true }, // colormap
                wgpu::BufferBindingType::Uniform,                     // int params
                wgpu::BufferBindingType::Uniform,                     // float params
            ],
            &[
                cs_table_buffer.as_entire_binding(),
                cs_value_buffer.as_entire_binding(),
                cs_position_buffer.as_entire_binding(),
                cs_normal_buffer.as_entire_binding(),
                cs_color_buffer.as_entire_binding(),
                cs_index_buffer.as_entire_binding(),
                cs_indirect_buffer.as_entire_binding(),
                cs_colormap_buffer.as_entire_binding(),
                cs_int_buffer.as_entire_binding(),
                cs_float_buffer.as_entire_binding(),
            ],
        );

        let cs_pipeline_layout =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&cs_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let cs_pipeline = init
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&cs_pipeline_layout),
                module: &cs_comp,
                entry_point: "cs_main",
            });

        Self {
            init,
            pipeline,
            uniform_bind_groups: vec![vert_bind_group, frag_bind_group],
            uniform_buffers: vec![
                vert_uniform_buffer,
                light_uniform_buffer,
                material_uniform_buffer,
            ],

            cs_pipelines: vec![cs_value_pipeline, cs_pipeline],
            cs_vertex_buffers: vec![
                cs_value_buffer,
                cs_position_buffer,
                cs_normal_buffer,
                cs_color_buffer,
            ],
            cs_index_buffer,
            cs_uniform_buffers: vec![
                cs_value_int_buffer,
                cs_value_float_buffer,
                cs_int_buffer,
                cs_float_buffer,
                cs_indirect_buffer,
            ],
            cs_bind_groups: vec![cs_value_bind_group, cs_bind_group],

            view_mat,
            project_mat,
            msaa_texture_view,
            depth_texture_view,

            animation_speed: 1.0,
            rotation_speed: 1.0,

            resolution: resol,
            index_count,
            surface_type: 2,
            colormap_direction: 1,
            colormap_reverse: 0,
            isolevel: 0.0,
            scale: 1.0,

            rng: rand::thread_rng(),
            t0: std::time::Instant::now(),
            random_shape_change: 0,
            fps_counter: ws::FpsCounter::default(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.init.size = new_size;
            self.init.config.width = new_size.width;
            self.init.config.height = new_size.height;
            self.init
                .surface
                .configure(&self.init.device, &self.init.config);

            self.project_mat =
                ws::create_projection_mat(new_size.width as f32 / new_size.height as f32, true);
            self.depth_texture_view = ws::create_depth_view(&self.init);
            if self.init.sample_count > 1 {
                self.msaa_texture_view = ws::create_msaa_texture_view(&self.init);
            }
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match keycode {
                VirtualKeyCode::Space => {
                    self.surface_type = (self.surface_type + 1) % 9;
                    println!(
                        "key = {:?}, surface_type = {:?}",
                        self.surface_type,
                        get_surface_type(self.surface_type)
                    );
                    true
                }
                VirtualKeyCode::LControl => {
                    self.colormap_direction = (self.colormap_direction + 1) % 4;
                    true
                }
                VirtualKeyCode::LShift => {
                    self.random_shape_change = (self.random_shape_change + 1) % 2;
                    true
                }
                VirtualKeyCode::LAlt => {
                    self.colormap_reverse = if self.colormap_reverse == 0 { 1 } else { 0 };
                    true
                }
                VirtualKeyCode::Q => {
                    self.animation_speed += 0.1;
                    true
                }
                VirtualKeyCode::A => {
                    self.animation_speed -= 0.1;
                    if self.animation_speed < 0.0 {
                        self.animation_speed = 0.0;
                    }
                    true
                }
                VirtualKeyCode::W => {
                    self.rotation_speed += 0.1;
                    true
                }
                VirtualKeyCode::S => {
                    self.rotation_speed -= 0.1;
                    if self.rotation_speed < 0.0 {
                        self.rotation_speed = 0.0;
                    }
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        // update uniform buffer
        let dt1 = self.rotation_speed * dt.as_secs_f32();

        let model_mat = ws::create_model_mat(
            [0.0, 0.0, 0.0],
            [dt1.sin(), dt1.cos(), 0.0],
            [1.0, 1.0, 1.0],
        );
        let view_project_mat = self.project_mat * self.view_mat;

        let normal_mat = (model_mat.invert().unwrap()).transpose();

        let model_ref: &[f32; 16] = model_mat.as_ref();
        let view_projection_ref: &[f32; 16] = view_project_mat.as_ref();
        let normal_ref: &[f32; 16] = normal_mat.as_ref();

        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 0, cast_slice(view_projection_ref));
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 64, cast_slice(model_ref));
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 128, cast_slice(normal_ref));

        let elapsed = self.t0.elapsed();
        if elapsed >= std::time::Duration::from_secs(5) && self.random_shape_change == 0 {
            self.surface_type = self.rng.gen_range(0..=8) as u32;
            self.t0 = std::time::Instant::now();
            println!(
                "key = {:?}, surface_type = {:?}",
                self.surface_type,
                get_surface_type(self.surface_type)
            );
        }

        // update compute buffers for value
        let value_int_params = [self.resolution, self.surface_type, 0, 0];
        self.init.queue.write_buffer(
            &self.cs_uniform_buffers[0],
            0,
            cast_slice(&value_int_params),
        );

        let value_float_params = [self.animation_speed * dt.as_secs_f32(), 0.0, 0.0, 0.0];
        self.init.queue.write_buffer(
            &self.cs_uniform_buffers[1],
            0,
            cast_slice(&value_float_params),
        );

        // update compute buffers for implicit surface
        let int_params = [
            self.resolution,
            self.surface_type,
            self.colormap_direction,
            self.colormap_reverse,
        ];
        self.init
            .queue
            .write_buffer(&self.cs_uniform_buffers[2], 0, cast_slice(&int_params));

        let float_params = [self.isolevel, self.scale, 0.0, 0.0];
        self.init
            .queue
            .write_buffer(&self.cs_uniform_buffers[3], 0, cast_slice(&float_params));

        let indirect_array = [500u32, 0, 0, 0];
        self.init
            .queue
            .write_buffer(&self.cs_uniform_buffers[4], 0, cast_slice(&indirect_array));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.init.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.init
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        // compute pass for value
        {
            let mut cs_index_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute value Pass"),
            });
            cs_index_pass.set_pipeline(&self.cs_pipelines[0]);
            cs_index_pass.set_bind_group(0, &self.cs_bind_groups[0], &[]);
            cs_index_pass.dispatch_workgroups(
                self.resolution / 8,
                self.resolution / 8,
                self.resolution / 8,
            );
        }

        // compute pass for vertices
        {
            let mut cs_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            cs_pass.set_pipeline(&self.cs_pipelines[1]);
            cs_pass.set_bind_group(0, &self.cs_bind_groups[1], &[]);
            cs_pass.dispatch_workgroups(
                self.resolution / 8,
                self.resolution / 8,
                self.resolution / 8,
            );
        }

        // render pass
        {
            let color_attach = ws::create_color_attachment(&view);
            let msaa_attach = ws::create_msaa_color_attachment(&view, &self.msaa_texture_view);
            let color_attachment = if self.init.sample_count == 1 {
                color_attach
            } else {
                msaa_attach
            };
            let depth_attachment = ws::create_depth_stencil_attachment(&self.depth_texture_view);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(depth_attachment),
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.cs_vertex_buffers[1].slice(..));
            render_pass.set_vertex_buffer(1, self.cs_vertex_buffers[2].slice(..));
            render_pass.set_vertex_buffer(2, self.cs_vertex_buffers[3].slice(..));
            render_pass.set_index_buffer(self.cs_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.uniform_bind_groups[0], &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_groups[1], &[]);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        self.fps_counter.print_fps(5);
        self.init.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    let mut sample_count = 1u32;
    let mut resolution = 192u32;
    let mut colormap_name = "jet";

    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }
    if args.len() > 2 {
        resolution = args[2].parse::<u32>().unwrap();
    }
    if args.len() > 3 {
        colormap_name = &args[3];
    }

    env_logger::init();
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    window.set_title(&*format!("{}", "implict_surface"));

    let mut state =
        pollster::block_on(State::new(&window, sample_count, resolution, colormap_name));
    let render_start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(_) => {
            let now = std::time::Instant::now();
            let dt = now - render_start_time;
            state.update(dt);

            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.init.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}