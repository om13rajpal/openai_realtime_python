# Flutter Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect Flutter app to FastAPI bridge with Siri-style voice UI redesign

**Architecture:** Replace direct OpenAI connection with bridge service. Add token caching, retry logic, and new Siri-style UI with animated wave, streaming text, and history drawer.

**Tech Stack:** Flutter, Dart, GetX, Lottie, flutter_dotenv, web_socket_channel

---

## Task 1: Update Environment Configuration

**Files:**
- Modify: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/.env`
- Modify: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/main.dart`

**Step 1: Update .env file with bridge URL**

```env
# Bridge Server Configuration
BRIDGE_URL=http://localhost:8000

# Environment (dev, staging, prod)
ENVIRONMENT=dev
```

**Step 2: Verify dotenv is loaded in main.dart**

Check that `main.dart` loads dotenv. If not, add:

```dart
import 'package:flutter_dotenv/flutter_dotenv.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(fileName: ".env");
  // ... rest of initialization
  runApp(const MyApp());
}
```

**Step 3: Test environment loads**

Run: `flutter run` and verify no errors loading .env

**Step 4: Commit**

```bash
git add .env lib/main.dart
git commit -m "feat: add bridge URL to environment config"
```

---

## Task 2: Create Bridge Events Model

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/services/voice/bridge_events.dart`

**Step 1: Write the bridge events model**

```dart
/// Bridge Server Event Models
///
/// Simplified event types for communication with the FastAPI bridge server.
/// The bridge handles all OpenAI complexity, exposing a cleaner protocol.
library;

import 'dart:convert';
import 'dart:typed_data';

/// Connection states for the bridge WebSocket connection.
enum BridgeConnectionState {
  disconnected,
  connecting,
  connected,
  reconnecting,
  error,
}

/// Voice UI states for the Siri-style interface.
enum VoiceUIState {
  /// Wave at bottom, idle
  idle,
  /// Wave animating to center, listening
  listening,
  /// Wave small, AI speaking with streaming text
  aiSpeaking,
  /// Wave medium, re-activated with history visible
  reactivated,
  /// Error state, wave paused
  error,
}

/// Message types sent from Flutter to Bridge.
enum OutgoingMessageType {
  audio('audio'),
  commit('commit'),
  cancel('cancel'),
  close('close');

  final String value;
  const OutgoingMessageType(this.value);
}

/// Message types received from Bridge.
enum IncomingMessageType {
  transcript('transcript'),
  audio('audio'),
  status('status'),
  error('error');

  final String value;
  const IncomingMessageType(this.value);

  static IncomingMessageType? fromString(String value) {
    for (final type in values) {
      if (type.value == value) return type;
    }
    return null;
  }
}

/// Status values from the bridge.
enum BridgeStatus {
  listening('listening'),
  speaking('speaking'),
  idle('idle');

  final String value;
  const BridgeStatus(this.value);

  static BridgeStatus? fromString(String value) {
    for (final status in values) {
      if (status.value == value) return status;
    }
    return null;
  }
}

/// Outgoing message to bridge server.
class BridgeOutgoingMessage {
  final OutgoingMessageType type;
  final Map<String, dynamic>? data;

  const BridgeOutgoingMessage({
    required this.type,
    this.data,
  });

  /// Create audio message with base64 encoded data.
  factory BridgeOutgoingMessage.audio(String base64Audio) {
    return BridgeOutgoingMessage(
      type: OutgoingMessageType.audio,
      data: {'data': base64Audio},
    );
  }

  /// Create commit message to force end of speech.
  factory BridgeOutgoingMessage.commit() {
    return const BridgeOutgoingMessage(type: OutgoingMessageType.commit);
  }

  /// Create cancel message to stop AI mid-response.
  factory BridgeOutgoingMessage.cancel() {
    return const BridgeOutgoingMessage(type: OutgoingMessageType.cancel);
  }

  /// Create close message to end session.
  factory BridgeOutgoingMessage.close() {
    return const BridgeOutgoingMessage(type: OutgoingMessageType.close);
  }

  String toJson() {
    final map = <String, dynamic>{'type': type.value};
    if (data != null) {
      map.addAll(data!);
    }
    return jsonEncode(map);
  }
}

/// Incoming message from bridge server.
class BridgeIncomingMessage {
  final IncomingMessageType type;
  final Map<String, dynamic> data;

  const BridgeIncomingMessage({
    required this.type,
    required this.data,
  });

  factory BridgeIncomingMessage.fromJson(String jsonStr) {
    final map = jsonDecode(jsonStr) as Map<String, dynamic>;
    final typeStr = map['type'] as String? ?? 'error';
    final type = IncomingMessageType.fromString(typeStr) ?? IncomingMessageType.error;
    return BridgeIncomingMessage(type: type, data: map);
  }

  /// Get transcript text and role.
  ({String text, String role})? get transcript {
    if (type != IncomingMessageType.transcript) return null;
    return (
      text: data['text'] as String? ?? '',
      role: data['role'] as String? ?? 'assistant',
    );
  }

  /// Get audio data as bytes.
  Uint8List? get audioData {
    if (type != IncomingMessageType.audio) return null;
    final base64 = data['data'] as String?;
    if (base64 == null) return null;
    return Uint8List.fromList(base64Decode(base64));
  }

  /// Get status value.
  BridgeStatus? get status {
    if (type != IncomingMessageType.status) return null;
    final statusStr = data['status'] as String?;
    if (statusStr == null) return null;
    return BridgeStatus.fromString(statusStr);
  }

  /// Get error details.
  ({String message, String? code})? get error {
    if (type != IncomingMessageType.error) return null;
    return (
      message: data['message'] as String? ?? 'Unknown error',
      code: data['code'] as String?,
    );
  }
}

/// Session response from POST /session.
class BridgeSessionResponse {
  final String token;
  final String conversationId;

  const BridgeSessionResponse({
    required this.token,
    required this.conversationId,
  });

  factory BridgeSessionResponse.fromJson(Map<String, dynamic> json) {
    return BridgeSessionResponse(
      token: json['token'] as String,
      conversationId: json['conversation_id'] as String,
    );
  }
}

/// Transcript entry for history.
class TranscriptEntry {
  final String role;
  final String text;
  final DateTime timestamp;

  const TranscriptEntry({
    required this.role,
    required this.text,
    required this.timestamp,
  });

  bool get isUser => role == 'user';
  bool get isAssistant => role == 'assistant';
}
```

**Step 2: Verify file compiles**

Run: `flutter analyze lib/services/voice/bridge_events.dart`
Expected: No errors

**Step 3: Commit**

```bash
git add lib/services/voice/bridge_events.dart
git commit -m "feat: add bridge events model for simplified protocol"
```

---

## Task 3: Create Session Manager

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/services/voice/session_manager.dart`

**Step 1: Write the session manager**

```dart
/// Session Manager for Bridge Server
///
/// Handles token caching, session creation, and conversation persistence.
/// Implements lazy loading with auto-refresh on token expiry.
library;

import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'bridge_events.dart';

class SessionManager {
  /// Cached session token
  String? _cachedToken;

  /// Cached conversation ID
  String? _cachedConversationId;

  /// When the token was obtained
  DateTime? _tokenTimestamp;

  /// Token validity duration (slightly less than server's 5 min TTL)
  static const _tokenValidityDuration = Duration(minutes: 4, seconds: 30);

  /// Whether a previous conversation exists that can be resumed
  bool _hasResumableConversation = false;

  /// Get the bridge base URL from environment.
  String get _baseUrl {
    final env = dotenv.env['ENVIRONMENT'] ?? 'dev';
    switch (env) {
      case 'prod':
        return 'https://api.aiseasafe.com';
      case 'staging':
        return 'https://staging-api.aiseasafe.com';
      default:
        return dotenv.env['BRIDGE_URL'] ?? 'http://localhost:8000';
    }
  }

  /// Get WebSocket URL for bridge connection.
  String get wsUrl {
    final base = _baseUrl.replaceFirst('http', 'ws');
    return '$base/ws?token=$_cachedToken';
  }

  /// Check if we have a valid cached token.
  bool get hasValidToken {
    if (_cachedToken == null || _tokenTimestamp == null) return false;
    final elapsed = DateTime.now().difference(_tokenTimestamp!);
    return elapsed < _tokenValidityDuration;
  }

  /// Check if a previous conversation can be resumed.
  bool get canResumeConversation => _hasResumableConversation && hasValidToken;

  /// Get current conversation ID.
  String? get conversationId => _cachedConversationId;

  /// Get or create a session token.
  ///
  /// Returns the cached token if valid, otherwise creates a new session.
  /// Set [forceNew] to true to always create a new conversation.
  Future<String> getToken({bool forceNew = false}) async {
    if (hasValidToken && !forceNew) {
      return _cachedToken!;
    }

    final response = await _createSession(
      conversationId: forceNew ? null : _cachedConversationId,
    );

    _cachedToken = response.token;
    _cachedConversationId = response.conversationId;
    _tokenTimestamp = DateTime.now();
    _hasResumableConversation = !forceNew && _cachedConversationId != null;

    return _cachedToken!;
  }

  /// Create a new session on the bridge server.
  Future<BridgeSessionResponse> _createSession({String? conversationId}) async {
    final uri = Uri.parse('$_baseUrl/session');

    final body = <String, dynamic>{
      'voice': 'alloy',
      'instructions': _getInstructions(),
    };

    if (conversationId != null) {
      body['conversation_id'] = conversationId;
    }

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );

    if (response.statusCode != 200) {
      throw SessionException(
        'Failed to create session: ${response.statusCode}',
        code: 'SESSION_CREATE_FAILED',
      );
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return BridgeSessionResponse.fromJson(json);
  }

  /// Get the system instructions for the AI.
  String _getInstructions() {
    // Maritime safety focused instructions (simplified for bridge)
    return '''You are an expert maritime weather assistant for recreational boating safety.
Keep voice responses concise and clear for audio delivery.
Focus on conditions relevant to recreational boaters: wind, waves, visibility.
Always err on the side of safety and caution.''';
  }

  /// Clear the cached session (e.g., on error or logout).
  void clearSession() {
    _cachedToken = null;
    _cachedConversationId = null;
    _tokenTimestamp = null;
    _hasResumableConversation = false;
  }

  /// Mark that a conversation exists for potential resume.
  void markConversationActive() {
    _hasResumableConversation = true;
  }

  /// Start a completely new conversation.
  Future<String> startNewConversation() async {
    _cachedConversationId = null;
    _hasResumableConversation = false;
    return getToken(forceNew: true);
  }

  /// Refresh the token (called when server rejects current token).
  Future<String> refreshToken() async {
    _cachedToken = null;
    _tokenTimestamp = null;
    return getToken();
  }
}

/// Exception for session-related errors.
class SessionException implements Exception {
  final String message;
  final String? code;

  const SessionException(this.message, {this.code});

  @override
  String toString() => 'SessionException: $message (code: $code)';
}
```

**Step 2: Add http dependency if not present**

Check `pubspec.yaml` for `http` package. If missing, add:

```yaml
dependencies:
  http: ^1.1.0
```

Then run: `flutter pub get`

**Step 3: Verify file compiles**

Run: `flutter analyze lib/services/voice/session_manager.dart`
Expected: No errors

**Step 4: Commit**

```bash
git add lib/services/voice/session_manager.dart pubspec.yaml
git commit -m "feat: add session manager with token caching and lazy loading"
```

---

## Task 4: Create Bridge Service

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/services/voice/bridge_service.dart`

**Step 1: Write the bridge service**

```dart
/// Bridge Service for FastAPI Server Connection
///
/// Handles WebSocket connection to the bridge server with:
/// - Auto-retry with exponential backoff
/// - Bidirectional message streaming
/// - Connection state management
library;

import 'dart:async';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'bridge_events.dart';
import 'session_manager.dart';

class BridgeService {
  final SessionManager _sessionManager;

  WebSocketChannel? _channel;
  StreamSubscription? _subscription;

  /// Current connection state
  BridgeConnectionState _connectionState = BridgeConnectionState.disconnected;
  BridgeConnectionState get connectionState => _connectionState;

  /// Stream controller for incoming messages
  final _messageController = StreamController<BridgeIncomingMessage>.broadcast();
  Stream<BridgeIncomingMessage> get messages => _messageController.stream;

  /// Stream controller for connection state changes
  final _stateController = StreamController<BridgeConnectionState>.broadcast();
  Stream<BridgeConnectionState> get stateChanges => _stateController.stream;

  /// Retry configuration
  static const _maxRetries = 3;
  static const _baseDelayMs = 1000;

  int _retryCount = 0;
  bool _isDisposed = false;

  BridgeService({SessionManager? sessionManager})
      : _sessionManager = sessionManager ?? SessionManager();

  /// Connect to the bridge server.
  ///
  /// Set [forceNew] to start a new conversation instead of resuming.
  Future<void> connect({bool forceNew = false}) async {
    if (_isDisposed) return;

    _updateState(BridgeConnectionState.connecting);

    try {
      final token = await _sessionManager.getToken(forceNew: forceNew);
      await _establishConnection();
      _retryCount = 0;
      _updateState(BridgeConnectionState.connected);
    } catch (e) {
      await _handleConnectionError(e, forceNew: forceNew);
    }
  }

  /// Establish WebSocket connection.
  Future<void> _establishConnection() async {
    final wsUrl = _sessionManager.wsUrl;
    _channel = WebSocketChannel.connect(Uri.parse(wsUrl));

    // Wait for connection to be ready
    await _channel!.ready;

    _subscription = _channel!.stream.listen(
      _onMessage,
      onError: _onError,
      onDone: _onDone,
    );
  }

  /// Handle incoming WebSocket message.
  void _onMessage(dynamic data) {
    if (_isDisposed) return;

    try {
      final message = BridgeIncomingMessage.fromJson(data as String);
      _messageController.add(message);

      // Mark conversation active on first successful message
      if (message.type == IncomingMessageType.status) {
        _sessionManager.markConversationActive();
      }
    } catch (e) {
      _messageController.addError(e);
    }
  }

  /// Handle WebSocket error.
  void _onError(dynamic error) {
    if (_isDisposed) return;
    _handleConnectionError(error);
  }

  /// Handle WebSocket close.
  void _onDone() {
    if (_isDisposed) return;

    if (_connectionState == BridgeConnectionState.connected) {
      // Unexpected disconnect, try to reconnect
      _handleConnectionError(Exception('Connection closed unexpectedly'));
    }
  }

  /// Handle connection errors with retry logic.
  Future<void> _handleConnectionError(dynamic error, {bool forceNew = false}) async {
    if (_isDisposed) return;

    _retryCount++;

    if (_retryCount <= _maxRetries) {
      _updateState(BridgeConnectionState.reconnecting);

      // Exponential backoff: 1s, 2s, 4s
      final delayMs = _baseDelayMs * (1 << (_retryCount - 1));
      await Future.delayed(Duration(milliseconds: delayMs));

      if (!_isDisposed) {
        // Try refreshing token on retry
        if (_retryCount > 1) {
          try {
            await _sessionManager.refreshToken();
          } catch (_) {}
        }
        await connect(forceNew: forceNew);
      }
    } else {
      _updateState(BridgeConnectionState.error);
      _messageController.addError(BridgeConnectionException(
        'Failed to connect after $_maxRetries retries',
        originalError: error,
      ));
    }
  }

  /// Update connection state and notify listeners.
  void _updateState(BridgeConnectionState state) {
    _connectionState = state;
    if (!_stateController.isClosed) {
      _stateController.add(state);
    }
  }

  /// Send audio data to the bridge.
  void sendAudio(String base64Audio) {
    _sendMessage(BridgeOutgoingMessage.audio(base64Audio));
  }

  /// Commit audio buffer (force end of speech).
  void commitAudio() {
    _sendMessage(BridgeOutgoingMessage.commit());
  }

  /// Cancel current AI response.
  void cancelResponse() {
    _sendMessage(BridgeOutgoingMessage.cancel());
  }

  /// Send a message to the bridge.
  void _sendMessage(BridgeOutgoingMessage message) {
    if (_channel == null || _connectionState != BridgeConnectionState.connected) {
      return;
    }

    try {
      _channel!.sink.add(message.toJson());
    } catch (e) {
      _messageController.addError(e);
    }
  }

  /// Disconnect from the bridge.
  Future<void> disconnect() async {
    if (_channel != null) {
      _sendMessage(BridgeOutgoingMessage.close());
      await _subscription?.cancel();
      await _channel?.sink.close();
      _channel = null;
      _subscription = null;
    }

    _updateState(BridgeConnectionState.disconnected);
  }

  /// Check if we can resume a previous conversation.
  bool get canResumeConversation => _sessionManager.canResumeConversation;

  /// Start a new conversation (clear history).
  Future<void> startNewConversation() async {
    await disconnect();
    await _sessionManager.startNewConversation();
    await connect(forceNew: true);
  }

  /// Dispose of resources.
  void dispose() {
    _isDisposed = true;
    disconnect();
    _messageController.close();
    _stateController.close();
  }
}

/// Exception for bridge connection errors.
class BridgeConnectionException implements Exception {
  final String message;
  final dynamic originalError;

  const BridgeConnectionException(this.message, {this.originalError});

  @override
  String toString() => 'BridgeConnectionException: $message';
}
```

**Step 2: Verify file compiles**

Run: `flutter analyze lib/services/voice/bridge_service.dart`
Expected: No errors

**Step 3: Commit**

```bash
git add lib/services/voice/bridge_service.dart
git commit -m "feat: add bridge service with WebSocket connection and retry logic"
```

---

## Task 5: Create Wave Animation Widget

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/widgets/voice/wave_animation.dart`

**Step 1: Write the wave animation widget**

```dart
/// Wave Animation Widget
///
/// Animated Lottie waveform that transitions between states:
/// - Idle: Small at bottom
/// - Listening: Large at center
/// - AI Speaking: Small at bottom
/// - Reactivated: Medium at bottom
library;

import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import '../../services/voice/bridge_events.dart';

class WaveAnimation extends StatefulWidget {
  final VoiceUIState state;
  final VoidCallback? onTap;
  final bool hasError;
  final String? errorMessage;

  const WaveAnimation({
    super.key,
    required this.state,
    this.onTap,
    this.hasError = false,
    this.errorMessage,
  });

  @override
  State<WaveAnimation> createState() => _WaveAnimationState();
}

class _WaveAnimationState extends State<WaveAnimation>
    with SingleTickerProviderStateMixin {
  late AnimationController _positionController;
  late Animation<double> _positionAnimation;
  late Animation<double> _scaleAnimation;

  static const _lottieUrl =
      'https://lottie.host/85e40b76-0c0e-45ae-a330-c436b197b24e/lDbtQSrqFQ.json';

  @override
  void initState() {
    super.initState();
    _positionController = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    _updateAnimations();
  }

  @override
  void didUpdateWidget(WaveAnimation oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.state != widget.state) {
      _updateAnimations();
      _positionController.forward(from: 0);
    }
  }

  void _updateAnimations() {
    final config = _getStateConfig(widget.state);

    _positionAnimation = Tween<double>(
      begin: _positionAnimation.value,
      end: config.verticalOffset,
    ).animate(CurvedAnimation(
      parent: _positionController,
      curve: Curves.easeOutCubic,
    ));

    _scaleAnimation = Tween<double>(
      begin: _scaleAnimation.value,
      end: config.scale,
    ).animate(CurvedAnimation(
      parent: _positionController,
      curve: Curves.easeOutCubic,
    ));
  }

  _WaveConfig _getStateConfig(VoiceUIState state) {
    switch (state) {
      case VoiceUIState.idle:
        return const _WaveConfig(verticalOffset: 0.35, scale: 0.6);
      case VoiceUIState.listening:
        return const _WaveConfig(verticalOffset: 0.0, scale: 1.0);
      case VoiceUIState.aiSpeaking:
        return const _WaveConfig(verticalOffset: 0.35, scale: 0.5);
      case VoiceUIState.reactivated:
        return const _WaveConfig(verticalOffset: 0.25, scale: 0.75);
      case VoiceUIState.error:
        return const _WaveConfig(verticalOffset: 0.35, scale: 0.6);
    }
  }

  @override
  void dispose() {
    _positionController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: widget.onTap,
      child: AnimatedBuilder(
        animation: _positionController,
        builder: (context, child) {
          return Stack(
            alignment: Alignment.center,
            children: [
              // Positioned wave
              Transform.translate(
                offset: Offset(
                  0,
                  MediaQuery.of(context).size.height * _positionAnimation.value,
                ),
                child: Transform.scale(
                  scale: _scaleAnimation.value,
                  child: ColorFiltered(
                    colorFilter: widget.hasError
                        ? const ColorFilter.mode(
                            Colors.grey,
                            BlendMode.saturation,
                          )
                        : const ColorFilter.mode(
                            Colors.transparent,
                            BlendMode.dst,
                          ),
                    child: Lottie.network(
                      _lottieUrl,
                      width: 300,
                      height: 150,
                      animate: !widget.hasError &&
                          widget.state != VoiceUIState.idle,
                    ),
                  ),
                ),
              ),

              // Error message
              if (widget.hasError && widget.errorMessage != null)
                Positioned(
                  bottom: 80,
                  child: Text(
                    widget.errorMessage!,
                    style: TextStyle(
                      color: Colors.grey[600],
                      fontSize: 14,
                    ),
                  ),
                ),

              // State hint text
              if (widget.state == VoiceUIState.listening)
                const Positioned(
                  top: 100,
                  child: Text(
                    'Listening...',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 16,
                      fontWeight: FontWeight.w300,
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

class _WaveConfig {
  final double verticalOffset;
  final double scale;

  const _WaveConfig({
    required this.verticalOffset,
    required this.scale,
  });
}
```

**Step 2: Create widgets/voice directory if needed**

```bash
mkdir -p lib/widgets/voice
```

**Step 3: Verify file compiles**

Run: `flutter analyze lib/widgets/voice/wave_animation.dart`
Expected: No errors

**Step 4: Commit**

```bash
git add lib/widgets/voice/wave_animation.dart
git commit -m "feat: add Lottie wave animation widget with state transitions"
```

---

## Task 6: Create Streaming Text Widget

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/widgets/voice/streaming_text.dart`

**Step 1: Write the streaming text widget**

```dart
/// Streaming Text Widget
///
/// Displays AI response text with typing animation effect.
/// Siri-style centered text that streams in character by character.
library;

import 'package:flutter/material.dart';

class StreamingText extends StatefulWidget {
  final String text;
  final bool isStreaming;
  final TextStyle? style;
  final TextAlign textAlign;

  const StreamingText({
    super.key,
    required this.text,
    this.isStreaming = false,
    this.style,
    this.textAlign = TextAlign.center,
  });

  @override
  State<StreamingText> createState() => _StreamingTextState();
}

class _StreamingTextState extends State<StreamingText>
    with SingleTickerProviderStateMixin {
  late AnimationController _cursorController;

  @override
  void initState() {
    super.initState();
    _cursorController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _cursorController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final defaultStyle = TextStyle(
      color: Colors.white,
      fontSize: 20,
      fontWeight: FontWeight.w400,
      height: 1.4,
    );

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: AnimatedSize(
        duration: const Duration(milliseconds: 200),
        child: RichText(
          textAlign: widget.textAlign,
          text: TextSpan(
            style: widget.style ?? defaultStyle,
            children: [
              TextSpan(text: widget.text),
              if (widget.isStreaming)
                WidgetSpan(
                  child: AnimatedBuilder(
                    animation: _cursorController,
                    builder: (context, child) {
                      return Opacity(
                        opacity: _cursorController.value,
                        child: Text(
                          '▌',
                          style: widget.style ?? defaultStyle,
                        ),
                      );
                    },
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Previous response display (shown above wave when reactivated).
class PreviousResponseText extends StatelessWidget {
  final String text;
  final int maxLines;

  const PreviousResponseText({
    super.key,
    required this.text,
    this.maxLines = 2,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Text(
        text,
        textAlign: TextAlign.center,
        maxLines: maxLines,
        overflow: TextOverflow.ellipsis,
        style: TextStyle(
          color: Colors.white54,
          fontSize: 16,
          fontWeight: FontWeight.w300,
        ),
      ),
    );
  }
}
```

**Step 2: Verify file compiles**

Run: `flutter analyze lib/widgets/voice/streaming_text.dart`
Expected: No errors

**Step 3: Commit**

```bash
git add lib/widgets/voice/streaming_text.dart
git commit -m "feat: add streaming text widget with cursor animation"
```

---

## Task 7: Create History Drawer Widget

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/widgets/voice/history_drawer.dart`

**Step 1: Write the history drawer widget**

```dart
/// History Drawer Widget
///
/// Swipe-up drawer showing conversation transcript history.
/// Simple list of You/AI exchanges with clean dividers.
library;

import 'package:flutter/material.dart';
import '../../services/voice/bridge_events.dart';

class HistoryDrawer extends StatelessWidget {
  final List<TranscriptEntry> history;
  final VoidCallback? onClose;
  final ScrollController? scrollController;

  const HistoryDrawer({
    super.key,
    required this.history,
    this.onClose,
    this.scrollController,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.95),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Handle bar
          GestureDetector(
            onTap: onClose,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 12),
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.white30,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
          ),

          // History list
          Flexible(
            child: history.isEmpty
                ? const _EmptyHistory()
                : ListView.separated(
                    controller: scrollController,
                    shrinkWrap: true,
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
                    itemCount: history.length,
                    separatorBuilder: (_, __) => const Divider(
                      color: Colors.white12,
                      height: 24,
                    ),
                    itemBuilder: (context, index) {
                      final entry = history[index];
                      return _HistoryEntry(entry: entry);
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _HistoryEntry extends StatelessWidget {
  final TranscriptEntry entry;

  const _HistoryEntry({required this.entry});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Role indicator
        Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: entry.isUser ? Colors.blue.withOpacity(0.2) : Colors.purple.withOpacity(0.2),
            shape: BoxShape.circle,
          ),
          child: Icon(
            entry.isUser ? Icons.person : Icons.assistant,
            size: 18,
            color: entry.isUser ? Colors.blue[300] : Colors.purple[300],
          ),
        ),
        const SizedBox(width: 12),

        // Text content
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                entry.isUser ? 'You' : 'AI',
                style: TextStyle(
                  color: Colors.white54,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                entry.text,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _EmptyHistory extends StatelessWidget {
  const _EmptyHistory();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.history,
            size: 48,
            color: Colors.white24,
          ),
          const SizedBox(height: 16),
          Text(
            'No conversation history',
            style: TextStyle(
              color: Colors.white38,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }
}
```

**Step 2: Verify file compiles**

Run: `flutter analyze lib/widgets/voice/history_drawer.dart`
Expected: No errors

**Step 3: Commit**

```bash
git add lib/widgets/voice/history_drawer.dart
git commit -m "feat: add history drawer widget with transcript list"
```

---

## Task 8: Create Voice Controller

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/controllers/voice_controller.dart`

**Step 1: Write the voice controller**

```dart
/// Voice Controller
///
/// GetX controller managing voice conversation state.
/// Coordinates bridge service, audio service, and UI state.
library;

import 'dart:async';
import 'package:get/get.dart';
import '../services/voice/bridge_service.dart';
import '../services/voice/bridge_events.dart';
import '../services/voice/audio_stream_service.dart';

class VoiceController extends GetxController {
  late final BridgeService _bridgeService;
  late final AudioStreamService _audioService;

  StreamSubscription? _messageSubscription;
  StreamSubscription? _stateSubscription;

  // Observable state
  final uiState = VoiceUIState.idle.obs;
  final connectionState = BridgeConnectionState.disconnected.obs;
  final currentText = ''.obs;
  final previousText = ''.obs;
  final isStreaming = false.obs;
  final hasError = false.obs;
  final errorMessage = RxnString();
  final history = <TranscriptEntry>[].obs;

  // Flags
  bool _isRecording = false;

  @override
  void onInit() {
    super.onInit();
    _bridgeService = BridgeService();
    _audioService = Get.find<AudioStreamService>();
    _setupListeners();
  }

  void _setupListeners() {
    _messageSubscription = _bridgeService.messages.listen(
      _handleMessage,
      onError: _handleError,
    );

    _stateSubscription = _bridgeService.stateChanges.listen((state) {
      connectionState.value = state;

      if (state == BridgeConnectionState.error) {
        hasError.value = true;
        uiState.value = VoiceUIState.error;
      } else if (state == BridgeConnectionState.connected) {
        hasError.value = false;
        errorMessage.value = null;
      }
    });
  }

  void _handleMessage(BridgeIncomingMessage message) {
    switch (message.type) {
      case IncomingMessageType.transcript:
        final transcript = message.transcript;
        if (transcript != null) {
          _handleTranscript(transcript.text, transcript.role);
        }
        break;

      case IncomingMessageType.audio:
        final audioData = message.audioData;
        if (audioData != null) {
          _audioService.playAudioChunk(audioData);
        }
        break;

      case IncomingMessageType.status:
        final status = message.status;
        if (status != null) {
          _handleStatus(status);
        }
        break;

      case IncomingMessageType.error:
        final error = message.error;
        if (error != null) {
          _handleError(Exception(error.message));
        }
        break;
    }
  }

  void _handleTranscript(String text, String role) {
    if (role == 'user') {
      // User transcript - add to history
      history.add(TranscriptEntry(
        role: 'user',
        text: text,
        timestamp: DateTime.now(),
      ));
    } else {
      // AI transcript - stream it
      currentText.value = text;
      isStreaming.value = true;

      // Add to history when complete (on status change)
    }
  }

  void _handleStatus(BridgeStatus status) {
    switch (status) {
      case BridgeStatus.listening:
        uiState.value = _isRecording && history.isNotEmpty
            ? VoiceUIState.reactivated
            : VoiceUIState.listening;
        isStreaming.value = false;
        break;

      case BridgeStatus.speaking:
        uiState.value = VoiceUIState.aiSpeaking;
        break;

      case BridgeStatus.idle:
        // AI finished speaking - finalize transcript
        if (currentText.value.isNotEmpty) {
          history.add(TranscriptEntry(
            role: 'assistant',
            text: currentText.value,
            timestamp: DateTime.now(),
          ));
          previousText.value = currentText.value;
          currentText.value = '';
        }
        isStreaming.value = false;
        uiState.value = VoiceUIState.idle;
        break;
    }
  }

  void _handleError(dynamic error) {
    hasError.value = true;
    errorMessage.value = error.toString();
    uiState.value = VoiceUIState.error;
    _stopRecording();
  }

  /// Toggle voice recording on/off.
  Future<void> toggleVoice() async {
    if (hasError.value) {
      // Clear error and retry
      hasError.value = false;
      errorMessage.value = null;
      await _bridgeService.connect();
      return;
    }

    if (_isRecording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (connectionState.value != BridgeConnectionState.connected) {
      await _bridgeService.connect();
    }

    _isRecording = true;
    uiState.value = history.isNotEmpty
        ? VoiceUIState.reactivated
        : VoiceUIState.listening;

    await _audioService.startRecording((audioData) {
      _bridgeService.sendAudio(audioData);
    });
  }

  Future<void> _stopRecording() async {
    _isRecording = false;
    await _audioService.stopRecording();
    _bridgeService.commitAudio();
  }

  /// Start a new conversation (clear history).
  Future<void> startNewConversation() async {
    history.clear();
    currentText.value = '';
    previousText.value = '';
    await _bridgeService.startNewConversation();
  }

  /// Resume the previous conversation.
  Future<void> resumeConversation() async {
    await _bridgeService.connect();
  }

  /// Check if a previous conversation can be resumed.
  bool get canResume => _bridgeService.canResumeConversation;

  /// Cancel current AI response.
  void cancelResponse() {
    _bridgeService.cancelResponse();
    _audioService.stopPlayback();
  }

  @override
  void onClose() {
    _messageSubscription?.cancel();
    _stateSubscription?.cancel();
    _bridgeService.dispose();
    super.onClose();
  }
}
```

**Step 2: Verify file compiles**

Run: `flutter analyze lib/controllers/voice_controller.dart`
Expected: No errors

**Step 3: Commit**

```bash
git add lib/controllers/voice_controller.dart
git commit -m "feat: add voice controller with GetX state management"
```

---

## Task 9: Create Voice Screen

**Files:**
- Create: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/screens/voice/voice_screen.dart`

**Step 1: Write the voice screen**

```dart
/// Voice Screen
///
/// Siri-style voice interface with:
/// - Animated wave at bottom (tap to activate)
/// - Streaming text in center
/// - Swipe-up history drawer
library;

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../controllers/voice_controller.dart';
import '../../services/voice/bridge_events.dart';
import '../../widgets/voice/wave_animation.dart';
import '../../widgets/voice/streaming_text.dart';
import '../../widgets/voice/history_drawer.dart';

class VoiceScreen extends StatefulWidget {
  const VoiceScreen({super.key});

  @override
  State<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen> {
  late VoiceController _controller;
  final DraggableScrollableController _drawerController =
      DraggableScrollableController();
  bool _showResumeOptions = false;

  @override
  void initState() {
    super.initState();
    _controller = Get.put(VoiceController());

    // Check if we should show resume options
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_controller.canResume) {
        setState(() => _showResumeOptions = true);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            // Main content
            _buildMainContent(),

            // History drawer
            _buildHistoryDrawer(),

            // Resume options overlay
            if (_showResumeOptions) _buildResumeOptions(),
          ],
        ),
      ),
    );
  }

  Widget _buildMainContent() {
    return Column(
      children: [
        // Close button
        Align(
          alignment: Alignment.topRight,
          child: IconButton(
            icon: const Icon(Icons.close, color: Colors.white54),
            onPressed: () => Get.back(),
          ),
        ),

        // Spacer
        const Spacer(),

        // Previous text (when reactivated)
        Obx(() {
          if (_controller.uiState.value == VoiceUIState.reactivated &&
              _controller.previousText.value.isNotEmpty) {
            return PreviousResponseText(text: _controller.previousText.value);
          }
          return const SizedBox.shrink();
        }),

        // Streaming text (during AI response)
        Obx(() {
          if (_controller.uiState.value == VoiceUIState.aiSpeaking &&
              _controller.currentText.value.isNotEmpty) {
            return StreamingText(
              text: _controller.currentText.value,
              isStreaming: _controller.isStreaming.value,
            );
          }
          return const SizedBox.shrink();
        }),

        // Spacer
        const Spacer(),

        // Wave animation
        Obx(() => WaveAnimation(
          state: _controller.uiState.value,
          onTap: _controller.toggleVoice,
          hasError: _controller.hasError.value,
          errorMessage: _controller.errorMessage.value,
        )),

        // Swipe hint
        Padding(
          padding: const EdgeInsets.only(bottom: 32),
          child: Text(
            'Swipe up for history',
            style: TextStyle(
              color: Colors.white24,
              fontSize: 12,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildHistoryDrawer() {
    return DraggableScrollableSheet(
      controller: _drawerController,
      initialChildSize: 0,
      minChildSize: 0,
      maxChildSize: 0.7,
      snap: true,
      snapSizes: const [0, 0.4, 0.7],
      builder: (context, scrollController) {
        return Obx(() => HistoryDrawer(
          history: _controller.history.toList(),
          scrollController: scrollController,
          onClose: () => _drawerController.animateTo(
            0,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
          ),
        ));
      },
    );
  }

  Widget _buildResumeOptions() {
    return Container(
      color: Colors.black87,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Previous conversation found',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 24),

            // Resume button
            ElevatedButton(
              onPressed: () {
                setState(() => _showResumeOptions = false);
                _controller.resumeConversation();
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                padding: const EdgeInsets.symmetric(
                  horizontal: 32,
                  vertical: 12,
                ),
              ),
              child: const Text('Resume conversation'),
            ),
            const SizedBox(height: 12),

            // New conversation button
            TextButton(
              onPressed: () {
                setState(() => _showResumeOptions = false);
                _controller.startNewConversation();
              },
              child: const Text(
                'Start new conversation',
                style: TextStyle(color: Colors.white54),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _drawerController.dispose();
    super.dispose();
  }
}
```

**Step 2: Create screens/voice directory if needed**

```bash
mkdir -p lib/screens/voice
```

**Step 3: Verify file compiles**

Run: `flutter analyze lib/screens/voice/voice_screen.dart`
Expected: No errors

**Step 4: Commit**

```bash
git add lib/screens/voice/voice_screen.dart
git commit -m "feat: add Siri-style voice screen with wave animation and history drawer"
```

---

## Task 10: Update Audio Stream Service

**Files:**
- Modify: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/services/voice/audio_stream_service.dart`

**Step 1: Add method stubs if missing**

Ensure `AudioStreamService` has these methods (add if missing):

```dart
/// Start recording and stream audio via callback.
Future<void> startRecording(void Function(String base64Audio) onAudioChunk);

/// Stop recording.
Future<void> stopRecording();

/// Play a chunk of audio data.
Future<void> playAudioChunk(Uint8List audioData);

/// Stop any ongoing playback.
Future<void> stopPlayback();
```

**Step 2: Verify existing implementation works**

The existing `AudioStreamService` should already handle most of this. Just ensure it:
- Encodes audio to base64 before callback
- Can play PCM16 audio chunks
- Is registered with GetX (`Get.put(AudioStreamService())`)

**Step 3: Commit if changes made**

```bash
git add lib/services/voice/audio_stream_service.dart
git commit -m "refactor: ensure audio service has required methods for bridge"
```

---

## Task 11: Wire Up Navigation

**Files:**
- Modify: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/routes/` (relevant route file)
- Modify: Entry point where voice is triggered

**Step 1: Add route for voice screen**

In the routes file, add:

```dart
GetPage(
  name: '/voice',
  page: () => const VoiceScreen(),
  binding: VoiceBinding(),
),
```

**Step 2: Create voice binding**

```dart
class VoiceBinding extends Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => AudioStreamService());
    Get.lazyPut(() => VoiceController());
  }
}
```

**Step 3: Update navigation calls**

Replace existing voice dialog navigation with:

```dart
Get.toNamed('/voice');
```

**Step 4: Test navigation**

Run the app and verify voice screen opens correctly.

**Step 5: Commit**

```bash
git add lib/routes/ lib/screens/voice/
git commit -m "feat: wire up voice screen navigation with GetX routing"
```

---

## Task 12: Delete Old Implementation

**Files:**
- Delete: `E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main/lib/services/voice/openai_realtime_service.dart`
- Modify: Any files importing the old service

**Step 1: Find all imports of old service**

```bash
grep -r "openai_realtime_service" lib/
```

**Step 2: Update imports to use bridge_service**

Replace:
```dart
import '../services/voice/openai_realtime_service.dart';
```

With:
```dart
import '../services/voice/bridge_service.dart';
```

**Step 3: Delete old service file**

```bash
rm lib/services/voice/openai_realtime_service.dart
```

**Step 4: Verify no broken imports**

Run: `flutter analyze`
Expected: No errors about missing imports

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove old OpenAI direct connection service"
```

---

## Task 13: Integration Testing

**Files:**
- All modified files

**Step 1: Start the FastAPI bridge server**

```bash
cd E:/aiseasafe
python -m uvicorn app.main:app --reload
```

**Step 2: Run Flutter app**

```bash
cd E:/agiready/Aiseasafe-Flutter-main/Aiseasafe-Flutter-main
flutter run
```

**Step 3: Test voice flow**

1. Navigate to voice screen
2. Tap wave to start listening
3. Speak a test phrase
4. Verify AI responds with audio and text
5. Swipe up to see history
6. Close and reopen to test resume flow

**Step 4: Test error handling**

1. Stop the bridge server
2. Try to use voice
3. Verify error message appears
4. Tap to retry after restarting server

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete Flutter integration with FastAPI bridge"
```

---

## Summary

| Task | Description | Complexity |
|------|-------------|------------|
| 1 | Update environment config | Low |
| 2 | Create bridge events model | Low |
| 3 | Create session manager | Medium |
| 4 | Create bridge service | Medium |
| 5 | Create wave animation widget | Medium |
| 6 | Create streaming text widget | Low |
| 7 | Create history drawer widget | Low |
| 8 | Create voice controller | Medium |
| 9 | Create voice screen | Medium |
| 10 | Update audio stream service | Low |
| 11 | Wire up navigation | Low |
| 12 | Delete old implementation | Low |
| 13 | Integration testing | Medium |

Total: 13 tasks
