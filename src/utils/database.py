import mysql.connector
from mysql.connector import Error

class MySQLDatabase:
    def __init__(self, host="localhost", user="root", password="123433", database="alphr_system_db"):
        """Khởi tạo cấu hình kết nối tới MySQL"""
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database
        }
        
    def insert_log(self, plate_text, image_path, vehicle_class, confidence):
        """Tự động chèn một bản ghi nhận diện xe mới vào bảng TrafficLogs"""
        connection = None
        cursor = None
        try:
            connection = mysql.connector.connect(**self.config)
            if connection.is_connected():
                cursor = connection.cursor()
                
                query = """
                INSERT INTO TrafficLogs (license_plate_text, image_path, vehicle_class, confidence)
                VALUES (%s, %s, %s, %s)
                """
                records = (plate_text, image_path, vehicle_class, confidence)
                
                cursor.execute(query, records)
                connection.commit()
                print(f" [DB SUCCESS] Đã lưu thành công biển số '{plate_text}' vào MySQL.")
        except Error as e:
            print(f"[DB ERROR] Lỗi khi kết nối hoặc chèn dữ liệu vào MySQL: {e}")
        
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None and connection.is_connected():
                connection.close()
                
    def test_connection(self) -> bool:
        try:
            conn = mysql.connector.connect(**self.config)
            ok   = conn.is_connected()
            conn.close()
            if ok:
                print("Kết nối MySQL thành công!")
            return ok
        except Error as e:
            print(f"Lỗi kết nối: {e}")
            return False