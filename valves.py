"""Solenoid valve control via Jetson.GPIO (non-blocking pulses)."""
import threading


class ValveController:
    def __init__(self, pins, pulse_seconds=0.05, active_high=True):
        import Jetson.GPIO as GPIO

        self.GPIO = GPIO
        self.pins = pins
        self.pulse = pulse_seconds
        self.on = GPIO.HIGH if active_high else GPIO.LOW
        self.off = GPIO.LOW if active_high else GPIO.HIGH
        GPIO.setmode(GPIO.BOARD)
        for pin in pins.values():
            GPIO.setup(pin, GPIO.OUT, initial=self.off)

    def fire(self, name):
        pin = self.pins.get(name)
        if pin is None:
            return
        self.GPIO.output(pin, self.on)
        threading.Timer(self.pulse, self.GPIO.output, args=(pin, self.off)).start()

    def cleanup(self):
        self.GPIO.cleanup()


class DummyValveController:
    """Used when --no-gpio is given (development off-device)."""

    def __init__(self, *args, **kwargs):
        pass

    def fire(self, name):
        print(f"[valve] {name}")

    def cleanup(self):
        pass
