import onnx
from onnx.tools import update_model_dims
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='INPUT ONNX file path'
    )
    parser.add_argument(
        '-of',
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='OUTPUT ONNX file path'
    )
    parser.add_argument(
        '-i',
        '--input_names',
        type=str,
        action='append',
        help='input names'
    )
    parser.add_argument(
        '-is',
        '--input_shapes',
        type=str,
        nargs='+',
        action='append',
        help='input shapes'
    )
    parser.add_argument(
        '-o',
        '--output_names',
        type=str,
        action='append',
        help='output names'
    )
    parser.add_argument(
        '-os',
        '--output_shapes',
        type=str,
        nargs='+',
        action='append',
        help='output shapes'
    )

    args = parser.parse_args()
    INPUT_MODEL_PATH = args.input_onnx_file_path
    OUTPUT_MODEL_PATH = args.output_onnx_file_path
    INPUT_NAMES = args.input_names
    INPUT_SHAPES = args.input_shapes
    OUTPUT_NAMES = args.output_names
    OUTPUT_SHAPES = args.output_shapes

    input_names = [name for name in INPUT_NAMES]
    input_shapes = [shape for shape in INPUT_SHAPES]
    output_names = [name for name in OUTPUT_NAMES]
    output_shapes = [shape for shape in OUTPUT_SHAPES]

    input_dicts = {name:shape for (name, shape) in zip(input_names, input_shapes)}
    output_dicts = {name:shape for (name, shape) in zip(output_names, output_shapes)}

    model = onnx.load(INPUT_MODEL_PATH)
    updated_model = update_model_dims.update_inputs_outputs_dims(
        model=model,
        input_dims=input_dicts,
        output_dims=output_dicts,
    )

    onnx.save(updated_model, OUTPUT_MODEL_PATH)