# ================================================================
# conexion.py
# ================================================================

import os
import oracledb
import keyring
from getpass import getpass
import sys

class OracleEnterpriseConnection:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.service_name = "oracle_enterprise_db"
    
    def setup_credentials(self):
        """
        Configura las credenciales en el keyring del sistema
        Solo necesitas ejecutar esto una vez
        """
        print("=== Configuración de Credenciales Oracle Enterprise ===")
        
        # Solicitar información de conexión
        username = input("Usuario Oracle: ")
        password = getpass("Password Oracle: ")
        host = input("Host Oracle (ej. prod-oracle.empresa.com): ")
        port = input("Puerto Oracle (1521): ") or "1521"
        sid = input("SID de la base de datos: ")
        
        # Almacenar en keyring
        keyring.set_password(f"{self.service_name}_user", "username", username)
        keyring.set_password(f"{self.service_name}_user", "password", password)
        keyring.set_password(f"{self.service_name}_connection", "host", host)
        keyring.set_password(f"{self.service_name}_connection", "port", port)
        keyring.set_password(f"{self.service_name}_connection", "sid", sid)
        
        print("✓ Credenciales almacenadas exitosamente en el keyring del sistema")
        print("⚠️  Las credenciales están seguras y solo tu usuario puede acceder a ellas")
    
    def get_credentials_from_keyring(self):
        """
        Recupera las credenciales del keyring del sistema
        """
        try:
            username = keyring.get_password(f"{self.service_name}_user", "username")
            password = keyring.get_password(f"{self.service_name}_user", "password")
            host = keyring.get_password(f"{self.service_name}_connection", "host")
            port = keyring.get_password(f"{self.service_name}_connection", "port")
            sid = keyring.get_password(f"{self.service_name}_connection", "sid")
            
            if not all([username, password, host, port, sid]):
                raise ValueError("Credenciales incompletas en keyring")
            
            return {
                'username': username,
                'password': password,
                'host': host,
                'port': port,
                'sid': sid
            }
        except Exception as e:
            print(f"✗ Error recuperando credenciales: {e}")
            return None
    
    def create_dsn_with_sid(self, host, port, sid):
        """
        Crea DSN para conexión con SID (Enterprise) usando oracledb
        """
        return f"{host}:{port}/{sid}"
    
    def connect(self):
        """
        Establece conexión con Oracle usando credenciales del keyring
        """
        try:
            # Obtener credenciales
            creds = self.get_credentials_from_keyring()
            if not creds:
                print("✗ No se encontraron credenciales. Ejecuta setup_credentials() primero")
                return False
            
            # Crear DSN con SID
            dsn = self.create_dsn_with_sid(
                creds['host'], 
                int(creds['port']), 
                creds['sid']
            )
            
            print(f"Conectando a: {creds['host']}:{creds['port']}/{creds['sid']}")
            
            # Establecer conexión con oracledb
            self.connection = oracledb.connect(
                user=creds['username'],
                password=creds['password'],
                dsn=dsn
            )
            
            self.cursor = self.connection.cursor()
            
            # Verificar conexión
            self.cursor.execute("SELECT SYSDATE, SYS_CONTEXT('USERENV', 'DB_NAME') FROM DUAL")
            result = self.cursor.fetchone()
            
            print("✓ Conexión exitosa a Oracle Enterprise")
            print(f"  - Fecha servidor: {result[0]}")
            print(f"  - Base de datos: {result[1]}")
            print(f"  - Usuario conectado: {creds['username']}")
            
            return True
            
        except oracledb.DatabaseError as e:
            print(f"✗ Error de Oracle: {e}")
            print("  Verifica:")
            print("  - Credenciales correctas")
            print("  - Host y puerto accesibles")
            print("  - SID correcto")
            print("  - Librerías Oracle disponibles")
            return False
        except Exception as e:
            print(f"✗ Error de conexión: {e}")
            return False
    
    def test_connection(self):
        """
        Realiza pruebas básicas de conectividad
        """
        if not self.cursor:
            print("✗ No hay conexión activa")
            return False
        
        try:
            print("\n=== Test de Conexión ===")
            
            # Test 1: Información del servidor
            self.cursor.execute("""
                SELECT 
                    BANNER,
                    CON_ID 
                FROM V$VERSION 
                WHERE ROWNUM = 1
            """)
            version_info = self.cursor.fetchone()
            print(f"✓ Versión Oracle: {version_info[0]}")
            
            # Test 2: Información de la sesión
            self.cursor.execute("""
                SELECT 
                    SYS_CONTEXT('USERENV', 'SESSION_USER') as usuario,
                    SYS_CONTEXT('USERENV', 'HOST') as host_cliente,
                    SYS_CONTEXT('USERENV', 'IP_ADDRESS') as ip_cliente,
                    SYS_CONTEXT('USERENV', 'SERVER_HOST') as host_servidor
                FROM DUAL
            """)
            session_info = self.cursor.fetchone()
            print(f"✓ Usuario: {session_info[0]}")
            print(f"✓ Host cliente: {session_info[1]}")
            print(f"✓ IP cliente: {session_info[2]}")
            print(f"✓ Host servidor: {session_info[3]}")
            
            # Test 3: Privilegios básicos
            self.cursor.execute("""
                SELECT COUNT(*) FROM USER_TABLES
            """)
            table_count = self.cursor.fetchone()[0]
            print(f"✓ Tablas accesibles: {table_count}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error en test: {e}")
            return False
    
    def execute_query(self, query, params=None, fetch_size=1000):
        """
        Ejecuta una consulta con parámetros opcionales
        """
        if not self.cursor:
            print("✗ No hay conexión activa")
            return None
        
        try:
            # Configurar fetch size para mejor rendimiento con oracledb
            self.cursor.arraysize = fetch_size
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT') or self.cursor.description:
                results = self.cursor.fetchall()
                print(f"✓ Consulta ejecutada: {len(results)} filas obtenidas")
                return results
            else:
                self.connection.commit()
                print(f"✓ Comando ejecutado: {self.cursor.rowcount} filas afectadas")
                return self.cursor.rowcount
                
        except Exception as e:
            print(f"✗ Error ejecutando consulta: {e}")
            self.connection.rollback()
            return None
    
    def get_table_info(self, table_name):
        """
        Obtiene información de una tabla
        """
        query = """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                DATA_LENGTH,
                NULLABLE,
                DATA_DEFAULT
            FROM USER_TAB_COLUMNS 
            WHERE TABLE_NAME = UPPER(:table_name)
            ORDER BY COLUMN_ID
        """
        return self.execute_query(query, {'table_name': table_name})
    
    def close_connection(self):
        """
        Cierra la conexión de forma segura
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            print("✓ Conexión cerrada correctamente")
        except Exception as e:
            print(f"⚠️  Error cerrando conexión: {e}")

def main():
    """
    Función principal para demostrar el uso
    """
    db = OracleEnterpriseConnection()
    
    # Verificar si es primera ejecución
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        db.setup_credentials()
        return
    
    # Conectar a la base de datos
    if not db.connect():
        print("\nPara configurar credenciales, ejecuta:")
        print("python conexion.py --setup")
        return
    
    # Realizar test de conexión
    db.test_connection()
    
    # Ejemplo de consulta
    print("\n=== Ejemplo de Consulta ===")
    results = db.execute_query("SELECT COUNT(*) FROM ALL_TABLES WHERE OWNER = USER")
    if results:
        print(f"Número de tablas del usuario: {results[0][0]}")
    
    # Cerrar conexión
    db.close_connection()

if __name__ == "__main__":
    main()