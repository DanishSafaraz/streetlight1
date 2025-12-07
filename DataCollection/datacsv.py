# datacsv.py
import paho.mqtt.client as mqtt
import ssl
import csv
import time
from datetime import datetime
import os
import sys

# =============================================
# KONFIGURASI
# =============================================
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "iot/streetlight/data"
MQTT_USERNAME = "hivemq.webclient.1765084748576"
MQTT_PASSWORD = "y9!uH7$Id8La4>PDCm.h"

CSV_FILENAME = "streetlight_data.csv"
TARGET_SAMPLES = 150  # Maksimal 150 data

# =============================================
# VARIABEL GLOBAL
# =============================================
data_count = 0
csv_file = None
csv_writer = None
start_time = None
client = None

# =============================================
# FUNGSI UTAMA
# =============================================

def setup_csv():
    """Setup file CSV untuk menyimpan data"""
    global csv_file, csv_writer
    
    print(f"📁 Creating CSV file: {CSV_FILENAME}")
    
    # Cek apakah file sudah ada
    if os.path.exists(CSV_FILENAME):
        print("⚠ File already exists! Creating backup...")
        backup_name = f"streetlight_data_backup_{int(time.time())}.csv"
        os.rename(CSV_FILENAME, backup_name)
        print(f"   Backup created: {backup_name}")
    
    # Buat file CSV baru
    csv_file = open(CSV_FILENAME, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    
    # Tulis header
    headers = ["timestamp", "light_intensity", "voltage"]
    csv_writer.writerow(headers)
    
    print(f"✅ CSV file ready. Waiting for data...")
    return csv_writer

def parse_mqtt_payload(payload):
    """Parse data dari format MQTT: {timestamp;intensity;voltage}"""
    try:
        # Hapus kurung kurawal
        payload = payload.strip('{}')
        
        # Split berdasarkan titik koma
        parts = payload.split(';')
        
        if len(parts) >= 3:
            timestamp = parts[0].strip()
            intensity = float(parts[1].strip())
            voltage = float(parts[2].strip())
            
            return {
                'timestamp': timestamp,
                'intensity': intensity,
                'voltage': voltage
            }
        else:
            print(f"⚠ Invalid format: {payload}")
            return None
            
    except Exception as e:
        print(f"❌ Error parsing payload: {e}")
        return None

def save_to_csv(data):
    """Simpan data ke CSV"""
    global csv_writer, data_count
    
    row = [
        data['timestamp'],
        data['intensity'],
        data['voltage']
    ]
    
    csv_writer.writerow(row)
    csv_file.flush()  # Pastikan data tersimpan
    data_count += 1
    
    return data_count

def show_progress():
    """Tampilkan progress bar"""
    progress = (data_count / TARGET_SAMPLES) * 100
    
    # Buat progress bar
    bar_length = 30
    filled = int(bar_length * data_count // TARGET_SAMPLES)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Hitung waktu yang sudah berlalu
    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    
    # Hitung rata-rata waktu per data
    if data_count > 0:
        avg_time = elapsed / data_count
        eta = (TARGET_SAMPLES - data_count) * avg_time
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
    else:
        eta_str = "--:--:--"
    
    print(f"\r📊 Progress: [{bar}] {data_count}/{TARGET_SAMPLES} ({progress:.1f}%) | "
          f"Time: {elapsed_str} | ETA: {eta_str}", end="")

# =============================================
# MQTT CALLBACKS
# =============================================

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback ketika terhubung ke MQTT"""
    if rc == 0:
        print("✅ Connected to MQTT broker!")
        print(f"📡 Subscribing to topic: {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ Connection failed with code: {rc}")

def on_message(client, userdata, msg):
    """Callback ketika menerima pesan"""
    global data_count
    
    # Jika sudah mencapai target, stop
    if data_count >= TARGET_SAMPLES:
        return
    
    try:
        payload = msg.payload.decode('utf-8')
        data = parse_mqtt_payload(payload)
        
        if data:
            # Simpan ke CSV
            save_to_csv(data)
            
            # Tampilkan data setiap 10 sampel
            if data_count % 10 == 0:
                print(f"\n📝 Data #{data_count}: {data['timestamp']} | "
                      f"Int: {data['intensity']}% | Volt: {data['voltage']}V")
            
            # Update progress bar
            show_progress()
            
            # Jika sudah mencapai target
            if data_count >= TARGET_SAMPLES:
                print(f"\n\n🎯 TARGET REACHED! {TARGET_SAMPLES} data collected!")
                finish_program()
                
    except Exception as e:
        print(f"\n❌ Error processing message: {e}")

def on_disconnect(client, userdata, rc, properties=None):
    """Callback ketika terputus dari MQTT"""
    if rc != 0:
        print("⚠ Disconnected from MQTT. Trying to reconnect...")

# =============================================
# FUNGSI PROGRAM UTAMA
# =============================================

def finish_program():
    """Selesaikan program dengan rapi"""
    global csv_file, client
    
    print("\n" + "="*50)
    print("📈 DATA COLLECTION COMPLETE!")
    print("="*50)
    
    # Tampilkan summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total data collected: {data_count}")
    print(f"   CSV file: {CSV_FILENAME}")
    print(f"   File size: {os.path.getsize(CSV_FILENAME)} bytes")
    
    # Tampilkan preview data
    print(f"\n📋 DATA PREVIEW (first 5 rows):")
    try:
        with open(CSV_FILENAME, 'r') as f:
            lines = f.readlines()[:6]  # Header + 5 data
            for line in lines:
                print(f"   {line.strip()}")
    except:
        print("   Could not read file for preview")
    
    # Tutup file
    if csv_file:
        csv_file.close()
        print(f"\n💾 CSV file saved successfully!")
    
    # Disconnect MQTT
    if client:
        client.disconnect()
        print("🔌 Disconnected from MQTT broker")
    
    print("\n👋 Program finished successfully!")
    print("="*50)
    
    sys.exit(0)

def main():
    """Fungsi utama program"""
    global client, start_time
    
    print("\n" + "="*50)
    print("      STREETLIGHT DATA COLLECTOR")
    print("="*50)
    print(f"🎯 Target: {TARGET_SAMPLES} data samples")
    print(f"💾 Output: {CSV_FILENAME}")
    print(f"📡 MQTT Server: {MQTT_BROKER}:{MQTT_PORT}")
    print("="*50 + "\n")
    
    # Setup CSV file
    setup_csv()
    
    # Setup MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    # Setup SSL/TLS
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    # Connect to broker
    print("🔗 Connecting to MQTT broker...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\n⚠ Troubleshooting tips:")
        print("   1. Check your internet connection")
        print("   2. Verify MQTT credentials")
        print("   3. Ensure Arduino is sending data")
        sys.exit(1)
    
    # Start timer
    start_time = time.time()
    
    # Start MQTT loop
    print("\n⏳ Listening for data...")
    print("   Press Ctrl+C to stop early")
    print("-"*50)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n\n⚠ Program interrupted by user")
        finish_program()

# =============================================
# RUN PROGRAM
# =============================================
if __name__ == "__main__":
    # Check if paho-mqtt is installed
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("❌ paho-mqtt library not installed!")
        print("   Install with: pip install paho-mqtt")
        sys.exit(1)
    
    # Run main program
    main()