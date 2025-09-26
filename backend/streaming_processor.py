import pandas as pd
import json
import time
import os
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
import traceback


class StreamingProcessor:
    """Multi-format streaming data processor for large files"""

    def __init__(self, file_path: str, chunk_size: int = 100000):
        """
        Initialize streaming processor

        Args:
            file_path: Path to the data file
            chunk_size: Number of rows to process per chunk
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    def stream_analysis(self) -> Dict[str, Any]:
        """Main streaming analysis - detects format and routes to appropriate method"""
        start_time = time.time()

        try:
            # Detect file format
            file_extension = self._get_file_extension()

            print(f"🚀 Streaming analysis started for {file_extension} format")
            print(f"📁 File size: {self._format_file_size(self.file_size)}")
            print(f"🔧 Chunk size: {self.chunk_size:,} rows")

            # Route to appropriate streaming method
            if file_extension == '.csv':
                result = self._stream_csv_analysis()
            elif file_extension == '.json':
                result = self._stream_json_analysis()
            elif file_extension in ['.xlsx', '.xls']:
                result = self._stream_excel_analysis()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Add common metadata
            result['file_info'] = {
                'file_path': self.file_path,
                'file_size_bytes': self.file_size,
                'file_size_formatted': self._format_file_size(self.file_size),
                'format': file_extension
            }

            total_duration = time.time() - start_time
            result['total_analysis_duration'] = total_duration

            print(f"✅ Streaming analysis completed in {total_duration:.2f}s")
            return result

        except Exception as e:
            error_msg = f"Streaming analysis error: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"📋 Error traceback: {traceback.format_exc()}")
            raise Exception(error_msg)

    def stream_json_analysis(self) -> Dict[str, Any]:
        """Public method for JSON streaming (backward compatibility)"""
        return self._stream_json_analysis()

    def stream_excel_analysis(self) -> Dict[str, Any]:
        """Public method for Excel streaming (backward compatibility)"""
        return self._stream_excel_analysis()

    def _get_file_extension(self) -> str:
        """Get file extension from file path"""
        _, ext = os.path.splitext(self.file_path.lower())
        return ext

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def _stream_csv_analysis(self) -> Dict[str, Any]:
        """CSV streaming analysis"""
        start_time = time.time()

        print("📊 Starting CSV streaming analysis...")

        # Initialize tracking variables
        total_rows = 0
        column_info = {}
        column_types = {}
        numeric_stats = {}
        categorical_stats = {}
        missing_values = {}
        data_quality = {
            'total_missing_cells': 0,
            'empty_rows': 0,
            'duplicate_rows': 0
        }

        chunk_count = 0
        error_chunks = 0

        try:
            # Process file in chunks
            for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size, low_memory=False):
                chunk_count += 1
                current_chunk_rows = len(chunk)
                total_rows += current_chunk_rows

                try:
                    # Analyze current chunk
                    self._analyze_chunk(
                        chunk, column_info, column_types, numeric_stats,
                        categorical_stats, missing_values, data_quality
                    )

                    print(f"  📄 Processed chunk {chunk_count}: {current_chunk_rows:,} rows")

                except Exception as chunk_error:
                    error_chunks += 1
                    print(f"  ⚠️ Error in chunk {chunk_count}: {chunk_error}")
                    continue

            # Finalize statistics
            self._finalize_statistics(numeric_stats, categorical_stats, total_rows)

            analysis_duration = time.time() - start_time

            # Prepare results
            result = {
                'method': 'streaming',
                'format': 'csv',
                'total_rows': total_rows,
                'total_columns': len(column_info),
                'column_info': column_info,
                'column_types': column_types,
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats,
                'missing_values': missing_values,
                'data_quality': data_quality,
                'performance': {
                    'analysis_duration': analysis_duration,
                    'chunk_size': self.chunk_size,
                    'chunks_processed': chunk_count,
                    'error_chunks': error_chunks,
                    'rows_per_second': total_rows / analysis_duration if analysis_duration > 0 else 0,
                    'mb_per_second': (self.file_size / (
                                1024 * 1024)) / analysis_duration if analysis_duration > 0 else 0
                }
            }

            print(f"✅ CSV analysis completed: {total_rows:,} rows, {len(column_info)} columns")
            return result

        except Exception as e:
            raise Exception(f"CSV streaming error: {str(e)}")

    def _stream_json_analysis(self) -> Dict[str, Any]:
        """JSON streaming analysis with JSON Lines support"""
        start_time = time.time()

        print("📊 Starting JSON streaming analysis...")

        try:
            # Detect JSON format type
            json_type = self._detect_json_type()
            print(f"📄 JSON type detected: {json_type}")

            if json_type == 'json_lines':
                return self._stream_json_lines_analysis()
            elif json_type == 'array':
                return self._stream_json_array_analysis()
            elif json_type == 'object':
                return self._stream_json_object_analysis()
            else:
                raise ValueError(f"Unsupported JSON structure: {json_type}")

        except Exception as e:
            raise Exception(f"JSON streaming error: {str(e)}")

    def _detect_json_type(self) -> str:
        """Detect JSON file structure"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to detect format
                first_lines = [f.readline().strip() for _ in range(min(5, 100))]
                f.seek(0)  # Reset file pointer

                # Check for JSON Lines format (each line is a JSON object)
                json_lines_count = 0
                for line in first_lines:
                    if line:
                        try:
                            json.loads(line)
                            json_lines_count += 1
                        except json.JSONDecodeError:
                            pass

                if json_lines_count >= 2:  # At least 2 valid JSON lines
                    return 'json_lines'

                # Check for regular JSON (array or object)
                try:
                    # Try to load first part to detect structure
                    sample = f.read(1024)  # Read first 1KB
                    f.seek(0)

                    # Parse sample to detect structure
                    if sample.strip().startswith('['):
                        return 'array'
                    elif sample.strip().startswith('{'):
                        return 'object'
                    else:
                        return 'unknown'

                except Exception:
                    return 'unknown'

        except Exception as e:
            print(f"⚠️ JSON type detection error: {e}")
            return 'unknown'

    def _stream_json_lines_analysis(self) -> Dict[str, Any]:
        """Stream JSON Lines format"""
        start_time = time.time()

        print("📊 Processing JSON Lines format...")

        # Initialize tracking
        total_rows = 0
        column_info = {}
        column_types = {}
        numeric_stats = {}
        categorical_stats = {}
        missing_values = {}
        data_quality = {'total_missing_cells': 0, 'empty_rows': 0, 'duplicate_rows': 0}

        chunk_count = 0
        error_lines = 0
        chunk_data = []

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse JSON line
                        data = json.loads(line)
                        chunk_data.append(data)

                        # Process chunk when it reaches chunk_size
                        if len(chunk_data) >= self.chunk_size:
                            chunk_count += 1
                            chunk_df = pd.DataFrame(chunk_data)
                            total_rows += len(chunk_df)

                            self._analyze_chunk(
                                chunk_df, column_info, column_types, numeric_stats,
                                categorical_stats, missing_values, data_quality
                            )

                            print(f"  📄 Processed JSON Lines chunk {chunk_count}: {len(chunk_df):,} rows")
                            chunk_data = []

                    except json.JSONDecodeError as e:
                        error_lines += 1
                        if error_lines <= 10:  # Log first 10 errors
                            print(f"  ⚠️ JSON parse error on line {line_num}: {e}")
                        continue

            # Process remaining data
            if chunk_data:
                chunk_count += 1
                chunk_df = pd.DataFrame(chunk_data)
                total_rows += len(chunk_df)

                self._analyze_chunk(
                    chunk_df, column_info, column_types, numeric_stats,
                    categorical_stats, missing_values, data_quality
                )

                print(f"  📄 Final JSON Lines chunk: {len(chunk_df):,} rows")

            # Finalize statistics
            self._finalize_statistics(numeric_stats, categorical_stats, total_rows)

            analysis_duration = time.time() - start_time

            result = {
                'method': 'streaming',
                'format': 'json_lines',
                'total_rows': total_rows,
                'total_columns': len(column_info),
                'column_info': column_info,
                'column_types': column_types,
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats,
                'missing_values': missing_values,
                'data_quality': data_quality,
                'json_info': {
                    'error_lines': error_lines,
                    'valid_lines': total_rows
                },
                'performance': {
                    'analysis_duration': analysis_duration,
                    'chunk_size': self.chunk_size,
                    'chunks_processed': chunk_count,
                    'error_lines': error_lines,
                    'rows_per_second': total_rows / analysis_duration if analysis_duration > 0 else 0,
                    'mb_per_second': (self.file_size / (
                                1024 * 1024)) / analysis_duration if analysis_duration > 0 else 0
                }
            }

            print(f"✅ JSON Lines analysis completed: {total_rows:,} rows, {error_lines} error lines")
            return result

        except Exception as e:
            raise Exception(f"JSON Lines streaming error: {str(e)}")

    def _stream_json_array_analysis(self) -> Dict[str, Any]:
        """Stream JSON array format"""
        start_time = time.time()

        print("📊 Processing JSON array format...")

        try:
            # Load entire JSON file (for arrays, we need to load all)
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Expected JSON array format")

            # Convert to DataFrame
            df = pd.DataFrame(data)
            total_rows = len(df)

            print(f"📄 Loaded JSON array with {total_rows:,} records")

            # Process in chunks
            return self._process_dataframe_in_chunks(df, 'json_array', start_time)

        except Exception as e:
            raise Exception(f"JSON array streaming error: {str(e)}")

    def _stream_json_object_analysis(self) -> Dict[str, Any]:
        """Stream single JSON object format"""
        start_time = time.time()

        print("📊 Processing JSON object format...")

        try:
            # Load JSON object
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Expected JSON object format")

            # Convert single object to DataFrame (1 row)
            df = pd.DataFrame([data])

            print(f"📄 Loaded single JSON object with {len(df.columns)} fields")

            # Process the single record
            return self._process_dataframe_in_chunks(df, 'json_object', start_time)

        except Exception as e:
            raise Exception(f"JSON object streaming error: {str(e)}")

    def _stream_excel_analysis(self) -> Dict[str, Any]:
        """Excel streaming analysis with multi-sheet support"""
        start_time = time.time()

        print("📊 Starting Excel streaming analysis...")

        try:
            # Get Excel file info
            excel_file = pd.ExcelFile(self.file_path)
            sheet_names = excel_file.sheet_names

            print(f"📄 Excel file contains {len(sheet_names)} sheets: {sheet_names}")

            # Process first sheet (or specified sheet)
            sheet_name = sheet_names[0]
            print(f"📄 Processing sheet: {sheet_name}")

            # Initialize tracking
            total_rows = 0
            column_info = {}
            column_types = {}
            numeric_stats = {}
            categorical_stats = {}
            missing_values = {}
            data_quality = {'total_missing_cells': 0, 'empty_rows': 0, 'duplicate_rows': 0}

            chunk_count = 0
            skip_rows = 0

            # Excel streaming - read in chunks
            while True:
                try:
                    # Read chunk from Excel
                    if skip_rows == 0:
                        # First chunk - include header
                        chunk = pd.read_excel(
                            self.file_path,
                            sheet_name=sheet_name,
                            nrows=self.chunk_size,
                            header=0
                        )
                    else:
                        # Subsequent chunks - skip header
                        chunk = pd.read_excel(
                            self.file_path,
                            sheet_name=sheet_name,
                            skiprows=skip_rows + 1,  # +1 to skip header
                            nrows=self.chunk_size,
                            header=None
                        )

                        # Set column names from first chunk
                        if hasattr(self, '_excel_columns'):
                            chunk.columns = self._excel_columns

                    # Check if chunk is empty
                    if chunk.empty or len(chunk) == 0:
                        break

                    # Store column names from first chunk
                    if not hasattr(self, '_excel_columns'):
                        self._excel_columns = chunk.columns.tolist()

                    chunk_count += 1
                    current_rows = len(chunk)
                    total_rows += current_rows

                    # Analyze chunk
                    self._analyze_chunk(
                        chunk, column_info, column_types, numeric_stats,
                        categorical_stats, missing_values, data_quality
                    )

                    print(f"  📄 Processed Excel chunk {chunk_count}: {current_rows:,} rows")

                    # If chunk is smaller than expected, we've reached the end
                    if len(chunk) < self.chunk_size:
                        break

                    skip_rows += self.chunk_size

                except Exception as chunk_error:
                    print(f"  ⚠️ Error processing Excel chunk {chunk_count}: {chunk_error}")
                    break

            # Clean up temporary attribute
            if hasattr(self, '_excel_columns'):
                delattr(self, '_excel_columns')

            # Finalize statistics
            self._finalize_statistics(numeric_stats, categorical_stats, total_rows)

            analysis_duration = time.time() - start_time

            result = {
                'method': 'streaming',
                'format': 'excel',
                'sheet_name': sheet_name,
                'available_sheets': sheet_names,
                'total_rows': total_rows,
                'total_columns': len(column_info),
                'column_info': column_info,
                'column_types': column_types,
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats,
                'missing_values': missing_values,
                'data_quality': data_quality,
                'excel_info': {
                    'active_sheet': sheet_name,
                    'total_sheets': len(sheet_names),
                    'all_sheets': sheet_names
                },
                'performance': {
                    'analysis_duration': analysis_duration,
                    'chunk_size': self.chunk_size,
                    'chunks_processed': chunk_count,
                    'rows_per_second': total_rows / analysis_duration if analysis_duration > 0 else 0,
                    'mb_per_second': (self.file_size / (
                                1024 * 1024)) / analysis_duration if analysis_duration > 0 else 0
                }
            }

            print(f"✅ Excel analysis completed: {total_rows:,} rows from sheet '{sheet_name}'")
            return result

        except Exception as e:
            raise Exception(f"Excel streaming error: {str(e)}")

    def _process_dataframe_in_chunks(self, df: pd.DataFrame, format_name: str, start_time: float) -> Dict[str, Any]:
        """Process a DataFrame in chunks (for JSON array/object)"""
        total_rows = len(df)
        column_info = {}
        column_types = {}
        numeric_stats = {}
        categorical_stats = {}
        missing_values = {}
        data_quality = {'total_missing_cells': 0, 'empty_rows': 0, 'duplicate_rows': 0}

        chunk_count = 0

        # Process in chunks
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]
            chunk_count += 1

            self._analyze_chunk(
                chunk, column_info, column_types, numeric_stats,
                categorical_stats, missing_values, data_quality
            )

            print(f"  📄 Processed {format_name} chunk {chunk_count}: {len(chunk):,} rows")

        # Finalize statistics
        self._finalize_statistics(numeric_stats, categorical_stats, total_rows)

        analysis_duration = time.time() - start_time

        return {
            'method': 'streaming',
            'format': format_name,
            'total_rows': total_rows,
            'total_columns': len(column_info),
            'column_info': column_info,
            'column_types': column_types,
            'numeric_stats': numeric_stats,
            'categorical_stats': categorical_stats,
            'missing_values': missing_values,
            'data_quality': data_quality,
            'performance': {
                'analysis_duration': analysis_duration,
                'chunk_size': self.chunk_size,
                'chunks_processed': chunk_count,
                'rows_per_second': total_rows / analysis_duration if analysis_duration > 0 else 0,
                'mb_per_second': (self.file_size / (1024 * 1024)) / analysis_duration if analysis_duration > 0 else 0
            }
        }

    def _analyze_chunk(self, chunk: pd.DataFrame, column_info: Dict, column_types: Dict,
                       numeric_stats: Dict, categorical_stats: Dict, missing_values: Dict,
                       data_quality: Dict):
        """Analyze a single chunk of data"""

        # Data quality checks
        data_quality['empty_rows'] += chunk.isnull().all(axis=1).sum()
        data_quality['total_missing_cells'] += chunk.isnull().sum().sum()

        for col in chunk.columns:
            col_data = chunk[col]
            col_name = str(col)  # Ensure column name is string

            # Initialize column info
            if col_name not in column_info:
                column_info[col_name] = {
                    'dtype': str(col_data.dtype),
                    'non_null_count': 0,
                    'unique_values': set(),
                    'first_seen_values': []
                }
                column_types[col_name] = str(col_data.dtype)
                missing_values[col_name] = 0

            # Update column statistics
            non_null_data = col_data.dropna()
            column_info[col_name]['non_null_count'] += len(non_null_data)
            missing_values[col_name] += col_data.isnull().sum()

            # Sample unique values (limit to prevent memory issues)
            if len(column_info[col_name]['unique_values']) < 1000:
                unique_sample = non_null_data.unique()[:100]  # Sample first 100 unique values
                column_info[col_name]['unique_values'].update(map(str, unique_sample))

            # Store first few values for preview
            if len(column_info[col_name]['first_seen_values']) < 5:
                sample_values = non_null_data.head(5 - len(column_info[col_name]['first_seen_values']))
                column_info[col_name]['first_seen_values'].extend(map(str, sample_values))

            # Numeric statistics
            if pd.api.types.is_numeric_dtype(col_data):
                if col_name not in numeric_stats:
                    numeric_stats[col_name] = {
                        'min': float('inf'),
                        'max': float('-inf'),
                        'sum': 0.0,
                        'sum_squares': 0.0,
                        'count': 0
                    }

                if len(non_null_data) > 0:
                    col_min = float(non_null_data.min())
                    col_max = float(non_null_data.max())
                    col_sum = float(non_null_data.sum())
                    col_sum_squares = float((non_null_data ** 2).sum())

                    numeric_stats[col_name]['min'] = min(numeric_stats[col_name]['min'], col_min)
                    numeric_stats[col_name]['max'] = max(numeric_stats[col_name]['max'], col_max)
                    numeric_stats[col_name]['sum'] += col_sum
                    numeric_stats[col_name]['sum_squares'] += col_sum_squares
                    numeric_stats[col_name]['count'] += len(non_null_data)

            # Categorical statistics
            elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
                if col_name not in categorical_stats:
                    categorical_stats[col_name] = defaultdict(int)

                # Count values (limit to prevent memory issues)
                value_counts = non_null_data.value_counts().head(100)  # Top 100 values
                for value, count in value_counts.items():
                    categorical_stats[col_name][str(value)] += count

    def _finalize_statistics(self, numeric_stats: Dict, categorical_stats: Dict, total_rows: int):
        """Calculate final statistics after processing all chunks"""

        # Finalize numeric statistics
        for col, stats in numeric_stats.items():
            if stats['count'] > 0:
                stats['mean'] = stats['sum'] / stats['count']
                # Calculate standard deviation
                variance = (stats['sum_squares'] / stats['count']) - (stats['mean'] ** 2)
                stats['std'] = np.sqrt(max(0, variance))  # Ensure non-negative
            else:
                stats['mean'] = 0
                stats['std'] = 0

            # Clean up intermediate calculations
            del stats['sum_squares']

        # Convert categorical stats defaultdict to regular dict
        for col in categorical_stats:
            categorical_stats[col] = dict(categorical_stats[col])

    def get_analysis_summary(self, result: Dict[str, Any]) -> List[str]:
        """Generate human-readable analysis summary"""
        summary = []

        try:
            format_name = result.get('format', 'unknown').upper()
            total_rows = result.get('total_rows', 0)
            total_cols = result.get('total_columns', 0)
            duration = result.get('performance', {}).get('analysis_duration', 0)

            summary.append(f"📊 {format_name} streaming analysis completed")
            summary.append(f"📈 Dataset: {total_rows:,} rows × {total_cols} columns")
            summary.append(f"⏱️ Processing time: {duration:.2f} seconds")
            summary.append(f"🚀 Speed: {result.get('performance', {}).get('rows_per_second', 0):.0f} rows/sec")

            # Data quality insights
            missing_pct = (result.get('data_quality', {}).get('total_missing_cells', 0) /
                           (total_rows * total_cols) * 100) if total_rows > 0 and total_cols > 0 else 0

            if missing_pct < 1:
                summary.append("✅ Excellent data quality - minimal missing values")
            elif missing_pct < 5:
                summary.append(f"✅ Good data quality - {missing_pct:.1f}% missing values")
            elif missing_pct < 15:
                summary.append(f"⚠️ Moderate data quality - {missing_pct:.1f}% missing values")
            else:
                summary.append(f"🚨 Poor data quality - {missing_pct:.1f}% missing values")

            # Format-specific insights
            if result.get('format') == 'excel':
                sheet_info = result.get('excel_info', {})
                if sheet_info.get('total_sheets', 0) > 1:
                    summary.append(f"📄 Excel file contains {sheet_info['total_sheets']} sheets")

            elif result.get('format') == 'json_lines':
                json_info = result.get('json_info', {})
                error_lines = json_info.get('error_lines', 0)
                if error_lines > 0:
                    summary.append(f"⚠️ {error_lines} JSON parsing errors encountered")

        except Exception as e:
            summary.append(f"Error generating summary: {str(e)}")

        return summary


# Example usage and testing functions
def test_streaming_processor():
    """Test function for streaming processor"""

    # This would be used for testing different file types
    test_files = [
        ('test_data.csv', 'CSV'),
        ('test_data.json', 'JSON'),
        ('test_data.xlsx', 'Excel')
    ]

    for file_path, file_type in test_files:
        if os.path.exists(file_path):
            print(f"\n🧪 Testing {file_type} streaming...")

            try:
                processor = StreamingProcessor(file_path, chunk_size=1000)
                result = processor.stream_analysis()

                summary = processor.get_analysis_summary(result)
                for line in summary:
                    print(f"  {line}")

            except Exception as e:
                print(f"  ❌ Test failed: {e}")
        else:
            print(f"  ⚠️ Test file not found: {file_path}")


if __name__ == "__main__":
    # Run tests if script is executed directly
    test_streaming_processor()
